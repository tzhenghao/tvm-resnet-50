# Standard imports
import logging

# Third party imports
import numpy as np
from PIL import Image
from tvm.contrib.download import download_testdata

# TVM imports
from tvm.driver import tvmc

logger = logging.getLogger(__name__)


TUNING_LOG_FILE = "resnet50-v2-7-python-autotuner_records.json"
TARGET = "llvm"
COMPILED_PACKAGE_PATH = "resnet50-v2-7-tvm-python.tar"
enable_relay_stdout = False
INPUTS = np.load("../imagenet_cat.npz")

if __name__ == "__main__":
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
    result = tvmc.run(package, device="cpu", inputs={"data": img_data})

    logger.warning("Results:")
    print(result)

    from scipy.special import softmax

    scores = softmax(result.get_output("output_0"))  # type: ignore
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
