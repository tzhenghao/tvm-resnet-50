# Standard imports
import logging

# Third party imports
import numpy as np

# TVM imports
from tvm.driver import tvmc

logger = logging.getLogger(__name__)


TUNING_LOG_FILE = "resnet50-v2-7-python-autotuner_records.json"
TARGET = "llvm"
COMPILED_PACKAGE_PATH = "resnet50-v2-7-tvm-python.tar"
enable_relay_stdout = False
INPUTS = np.load("../imagenet_cat.npz")

if __name__ == "__main__":
    model = tvmc.load(
        "../assets/resnet50-v2-7.onnx",
        shape_dict={"data": [1, 3, 224, 224]},
    )  # Step 1: Load + shape_dict

    if enable_relay_stdout:
        logger.warning("RELAY:")
        model.summary()  # display Relay

    # tvmc tune --target "llvm" --output resnet50-v2-7-python-autotuner_records.json assets/resnet50-v2-7.onnx
    # logger.warning("Tuning the model...")
    # tvmc.tune(model, target=TARGET, tuning_records=TUNING_LOG_FILE)

    # tvmc compile --target "llvm" --tuning-records resnet50-v2-7-autotuner_records.json  --output resnet50-v2-7-tvm_autotuned.tar assets/resnet50-v2-7.onnx
    logger.warning("Compiling model...")
    package = tvmc.compile(
        model,
        target=TARGET,
        package_path=COMPILED_PACKAGE_PATH,
        # tuning_records=TUNING_LOG_FILE,
    )

    logger.warning("Running model...")
    result = tvmc.run(package, device="cpu", inputs={"data": INPUTS})

    logger.warning("Results:")
    print(result)
