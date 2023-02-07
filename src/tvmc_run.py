# Standard imports
import logging
from typing import Any

# Third party imports
import numpy as np
from PIL import Image
from pydantic.dataclasses import dataclass
from tvm.contrib.download import download_testdata
from tvm.driver import tvmc

logger = logging.getLogger(__name__)


TUNING_LOG_FILE = "resnet50-v2-7-python-autotuner_records.json"
TARGET = "llvm"
COMPILED_PACKAGE_PATH = "resnet50-v2-7-tvm-python.tar"
enable_relay_stdout = False

# ---------------------------------- TVMC  ----------------------------------


def preprocess_tvmc():
    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    # Resize it to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # ONNX expects NCHW input, so convert the array
    img_data = np.transpose(img_data, (2, 0, 1))

    # Normalize according to ImageNet
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_stddev = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (
            img_data[i, :, :] / 255 - imagenet_mean[i]
        ) / imagenet_stddev[i]

    # Add batch dimension
    img_data = np.expand_dims(norm_img_data, axis=0)
    return img_data


def postprocess_tvmc(result: dict[str, Any]):
    # Third party imports
    from scipy.special import softmax

    scores = softmax(result["output_0"])
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]

    # Download a list of labels
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as file:
        labels = [label.rstrip() for label in file]
        for rank in ranks[0:5]:
            print(
                "class='%s' with probability=%f" % (labels[rank], scores[rank])
            )


@dataclass
class TVMC:
    """
    This class wraps all TVMC interfaces.
    """

    def init_onnx(self, onnx_file: str, shape_dict: dict[str, Any]):
        self.load_onnx(onnx_file=onnx_file, shape_dict=shape_dict)

    def load_onnx(self, onnx_file: str, shape_dict: dict[str, Any]):
        self.model = tvmc.load(
            onnx_file,
            shape_dict=shape_dict,
        )

    def print_summary(self):
        """
        This method prints the summary of the Relay IR.
        """
        self.model.summary()

    def compile_model(self, target: str):
        """
        This method compiles the TVMCModel

        tvmc compile --target "llvm" --tuning-records
        resnet50-v2-7-autotuner_records.json  --output
        resnet50-v2-7-tvm_autotuned.tar assets/resnet50-v2-7.onnx
        """
        logger.warning("Compiling model...")
        self.package = tvmc.compile(
            self.model,
            target=target,
            package_path=COMPILED_PACKAGE_PATH,
            # tuning_records=self.tuning_records,
        )

    def tune_model(self, target: str):
        """
        tvmc tune --target "llvm" --output
        resnet50-v2-7-python-autotuner_records.json assets/resnet50-v2-7.onnx
        """
        logger.warning("Tuning the model...")
        # tvmc.tune(self.model, target=target, tuning_records=TUNING_LOG_FILE)
        self.tuning_records = tvmc.tune(self.model, target=target)

    def run_model(self) -> dict[str, Any] | None:
        """
        This method runs the TVM model.
        """
        logger.warning("Running model...")
        result = tvmc.run(
            self.package, device="cpu", inputs={"data": img_data}
        )

        if result is None:
            logger.error("Expected result to be TVMCResult, got None instead")
            return result

        return result.outputs


if __name__ == "__main__":
    img_data = preprocess_tvmc()

    tvmc_instance = TVMC()

    INPUT_NAME = "data"
    shape_dict = {INPUT_NAME: img_data.shape}

    tvmc_instance.init_onnx(
        onnx_file="../assets/resnet50-v2-7.onnx",
        shape_dict=shape_dict,
    )

    if enable_relay_stdout:
        tvmc_instance.print_summary()

    tvmc_instance.compile_model(target=TARGET)

    result = tvmc_instance.run_model()

    if result is None:
        logger.warning("Results:")
        print(result)
    else:
        postprocess_tvmc(result=result)
