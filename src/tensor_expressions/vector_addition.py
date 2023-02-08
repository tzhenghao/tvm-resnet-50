# Standard imports
import logging

# Third party imports
import click
import numpy as np
import tvm
import tvm.testing
from tvm import te

logger = logging.getLogger(__name__)


TUNING_LOG_FILE = "resnet-50-v2-autotuning.json"
TARGET = "llvm"
COMPILED_PACKAGE_PATH = "resnet50-v2-7-tvm-python.tar"
enable_relay_stdout = False

if __name__ == "__main__":
    tgt = tvm.target.Target(target=TARGET, host=TARGET)

    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

    s = te.create_schedule(C.op)  # type: ignore

    click.secho("BEFORE OPTIMIZATION:", fg="green", bold=True)
