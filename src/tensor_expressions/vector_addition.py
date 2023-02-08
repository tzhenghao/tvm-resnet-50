# Standard imports
import logging
import timeit

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

    click.secho("Running numpy...", fg="yellow")
    np_repeat = 100
    np_running_time = timeit.timeit(
        setup="import numpy\n"
        "n = 32768\n"
        'dtype = "float32"\n'
        "a = numpy.random.rand(n, 1).astype(dtype)\n"
        "b = numpy.random.rand(n, 1).astype(dtype)\n",
        stmt="answer = a + b",
        number=np_repeat,
    )
    click.secho(
        "Numpy running time: {}".format(np_running_time / np_repeat),
        fg="green",
        bold=True,
    )

    click.secho("Running TE...", fg="yellow")
    dev = tvm.device(tgt.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = fadd.time_evaluator(fadd.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    click.secho(
        "TE running time: {}".format(mean_time),
        fg="green",
        bold=True,
    )

    click.secho("Done!", fg="green", bold=True)
