# tvm-resnet-50

A library that deploys a Resnet-50 V2 ONNX model to TVM runtime.
Also showcases some of the tuning options available in TVM.

## Installation Requirements / Instructions

You'll need `docker` and `nvidia-docker` installed on your machine.

1. Run `nvidia-smi` to verify that you have the proper Nvidia drivers
installed and that your supported GPUs are listed there.
2. `docker compose run -it zheng_tvm_local_dev bash`

## Examples

Here's an example of the AutoTVM output:

![autotvm-sample-output](/images/autotvm_run_sample_output.png)

### Parallel Implementation IR

![sample-parallel-ir](/images/sample-parallel-ir.png)

### GPU Implementation IR

![sample-gpu-code](/images/sample-gpu-code.png)

### Manually optimizing the matrix multiplication IR

![sample-manual-matmul-1](/images/sample-manual-matmul-1.png)

![sample-manual-matmul-2](/images/sample-manual-matmul-2.png)
