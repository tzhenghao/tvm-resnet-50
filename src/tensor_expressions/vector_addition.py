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

    schedule = te.create_schedule(C.op)  # type: ignore

    fadd = tvm.build(schedule, [A, B, C], tgt, name="myadd")

    dev = tvm.device(tgt.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)  # type: ignore

    fadd(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    click.secho("Done!", fg="green", bold=True)
