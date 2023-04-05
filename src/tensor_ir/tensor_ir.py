# Standard imports
import logging
import sys
import click

# Third party imports
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

TARGET = "llvm"
NUM_MEASUREMENTS = 5
TUNING_OUTPUT_FILE = "matmul.log"
DEFAULT_RTOL = 1e-4


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


if __name__ == "__main__":
    ir_module = MyModule
    print(type(ir_module))
    print(ir_module.script())

    from tvm import te

    A = te.placeholder((8,), dtype="float32", name="A")
    B = te.compute((8,), lambda *i: A(*i) + 1.0, name="B")
    func = te.create_prim_func([A, B])
    ir_module_from_te = IRModule({"main": func})
    print(ir_module_from_te.script())

    mod = tvm.build(ir_module_from_te, target=TARGET)  # The module for CPU backends.
    print(type(mod))

    a = tvm.nd.array(np.arange(8).astype("float32"))
    b = tvm.nd.array(np.zeros((8,)).astype("float32"))
    mod(a, b)
    print(a)
    print(b)
