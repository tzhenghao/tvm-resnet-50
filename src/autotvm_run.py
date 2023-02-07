# Standard imports
import logging
from typing import Any

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


TUNING_LOG_FILE = "resnet50-v2-7-python-autotuner_records.json"
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


class AutoTVM:
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

    def compile_model(self, target: str):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(self.mod, target=target, params=self.params)

        dev = tvm.device(str(target), 0)
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

    def benchmark_model(self):
        # Standard imports
        import timeit

        timing_number = 10
        timing_repeat = 10
        unoptimized = (
            np.array(
                timeit.Timer(lambda: self.module.run()).repeat(
                    repeat=timing_repeat, number=timing_number
                )
            )
            * 1000
            / timing_number
        )
        unoptimized = {
            "mean": np.mean(unoptimized),
            "median": np.median(unoptimized),
            "std": np.std(unoptimized),
        }

        print(unoptimized)


if __name__ == "__main__":
    img_data = preprocess_autotvm()

    autotvm_instance = AutoTVM()

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

    autotvm_instance.compile_model(target=TARGET)

    tvm_output = autotvm_instance.run_model()

    autotvm_instance.benchmark_model()

    postprocess_autotvm(tvm_output=tvm_output)
