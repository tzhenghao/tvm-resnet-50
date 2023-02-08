# Standard imports
import logging
from typing import Any
import click

# Third party imports
import numpy as np
import onnx
import tvm
import tvm.auto_scheduler as auto_scheduler
import tvm.relay as relay
from PIL import Image
from pydantic.dataclasses import dataclass
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
from tvm.driver import tvmc

logger = logging.getLogger(__name__)


TUNING_LOG_FILE = "resnet-50-v2-autotuning.json"
TARGET = "llvm"
COMPILED_PACKAGE_PATH = "resnet50-v2-7-tvm-python.tar"
enable_relay_stdout = False


# --------------------------------- AutoTVM  ---------------------------------
class ONNXLoader:
    def load(self, file):
        """
        This class method loads the ONNX model as specified by the file.
        """
        with open(file, "rb") as f:
            onnx_model = onnx.load(f)
        return onnx_model


def preprocess_autotvm():
    # Seed numpy's RNG to get consistent results
    np.random.seed(0)

    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    # Resize it to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # Our input image is in HWC layout while ONNX expects CHW input,
    # so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))

    # Normalize according to the ImageNet input specification
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(norm_img_data, axis=0)
    return img_data


def postprocess_autotvm(tvm_output):
    # Third party imports
    from scipy.special import softmax

    # Download a list of labels
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as file:
        labels = [label.rstrip() for label in file]

    # Open the output and read the output tensor
    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:5]:
        print("class='%s' with probability=%f" % (labels[rank], scores[rank]))


@dataclass
class AutoTVM:
    target: str
    # mod : tvm.IRModule
    #     The relay module for compilation
    # params : dict of str to tvm.nd.NDArray
    #     The parameter dict to be used by relay

    # TODO(zheng): Add more init_from_*(...)
    def init_from_onnx(self, onnx_model):
        """
        Initialize from an ONNX model.
        """
        self.mod, self.params = relay.frontend.from_onnx(
            onnx_model, shape_dict
        )

    def compile_model(self, tuning_records: str | None = None):
        """
        This method compiles the current model.

        Params
        -----
        tuning_records: str | None
            The str path to the tuning_records file.
        """
        if tuning_records is None:
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(
                    self.mod, target=self.target, params=self.params
                )

        else:
            with autotvm.apply_history_best(tuning_records):
                with tvm.transform.PassContext(opt_level=3, config={}):
                    lib = relay.build(
                        self.mod, target=self.target, params=self.params
                    )

        dev = tvm.device(str(self.target), 0)
        self.module = graph_executor.GraphModule(lib["default"](dev))

    def run_model(self):
        # dtype = "float32"
        self.module.set_input(INPUT_NAME, img_data)
        self.module.run()
        output_shape = (1, 1000)
        tvm_output = self.module.get_output(
            0, tvm.nd.empty(output_shape)
        ).numpy()
        return tvm_output

    def tune_model(
        self,
        num_configurations: int,
        num_measurements_per_config: int,
        min_ms_per_config: int,
        timeout: int,
    ):
        """
        Params
        -------
        num_configurations: int
            the number of different configurations that we will test
        num_measurements_per_config: int
            how many measurements we will take of each configuration.
        min_ms_per_config: int
            how long need to run configuration test. If the number of repeats
            falls under this time, it will be increased. This option is
            necessary for accurate tuning on GPUs, and is not required for
            CPU tuning. Setting this value to 0 disables it.
        timeout: int
            an upper limit on how long to run training code for each tested
            configuration
        """
        click.secho("Tuning the model...", fg="green", bold=True)
        # number = 10
        # repeat = 1
        # min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
        # timeout = 10  # in seconds

        # create a TVM runner
        runner = autotvm.LocalRunner(
            number=num_configurations,
            repeat=num_measurements_per_config,
            timeout=timeout,
            min_repeat_ms=min_ms_per_config,
            enable_cpu_cache_flush=True,
        )

        # TODO(zheng): Make this a tuning config dataclass.
        tuning_option = {
            "tuner": "xgb",
            "trials": 20,
            "early_stopping": 100,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"),
                runner=runner,
            ),
            "tuning_records": TUNING_LOG_FILE,
        }

        # begin by extracting the tasks from the onnx model
        tasks = autotvm.task.extract_from_program(
            self.mod["main"], target=self.target, params=self.params
        )

        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(task, loss_type="rank")
            tuner_obj.tune(
                n_trial=min(tuning_option["trials"], len(task.config_space)),
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=[
                    autotvm.callback.progress_bar(
                        tuning_option["trials"], prefix=prefix
                    ),
                    autotvm.callback.log_to_file(
                        tuning_option["tuning_records"]
                    ),
                ],
            )

    def benchmark_model(self):
        # Standard imports
        import timeit

        timing_number = 10
        timing_repeat = 10
        benchmark_results = (
            np.array(
                timeit.Timer(lambda: self.module.run()).repeat(
                    repeat=timing_repeat, number=timing_number
                )
            )
            * 1000
            / timing_number
        )
        benchmark_results = {
            "mean": np.mean(benchmark_results),
            "median": np.median(benchmark_results),
            "std": np.std(benchmark_results),
        }

        print(benchmark_results)


if __name__ == "__main__":
    img_data = preprocess_autotvm()

    autotvm_instance = AutoTVM(target=TARGET)

    INPUT_NAME = "data"
    shape_dict = {INPUT_NAME: img_data.shape}

    model_url = (
        "https://github.com/onnx/models/raw/main/"
        "vision/classification/resnet/model/"
        "resnet50-v2-7.onnx"
    )

    model_path = download_testdata(
        model_url, "resnet50-v2-7.onnx", module="onnx"
    )
    onnx_loader = ONNXLoader()
    onnx_model = onnx_loader.load(file=model_path)

    autotvm_instance.init_from_onnx(onnx_model=onnx_model)

    click.secho("BEFORE OPTIMIZATION:", fg="green", bold=True)

    autotvm_instance.compile_model()

    tvm_output = autotvm_instance.run_model()

    autotvm_instance.benchmark_model()

    postprocess_autotvm(tvm_output=tvm_output)

    click.secho("AFTER OPTIMIZATION:", fg="green", bold=True)

    # min_ms_per_config = 0  # since we're tuning on a CPU, can be set to 0
    # timeout = 10  # in seconds
    autotvm_instance.tune_model(
        num_configurations=10,
        num_measurements_per_config=1,
        min_ms_per_config=0,
        timeout=10,
    )

    autotvm_instance.compile_model(tuning_records=TUNING_LOG_FILE)

    tvm_output = autotvm_instance.run_model()

    autotvm_instance.benchmark_model()

    postprocess_autotvm(tvm_output=tvm_output)
