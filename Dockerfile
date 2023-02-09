FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# WORKDIR
ENV APP_HOME=/app
RUN mkdir -p $APP_HOME

# Set the working directory in the container
WORKDIR $APP_HOME

# Install basic packages.
RUN apt-get update \
    && apt-get -y install git python3.10-dev python3-pip \
    python3-setuptools libtinfo-dev zlib1g-dev build-essential \
    cmake ninja-build libedit-dev libxml2-dev vim \
    && apt-get clean

# Build LLVM
COPY infra/install_llvm.sh /infra/install_llvm.sh
RUN bash /infra/install_llvm.sh

# Build TVM
COPY infra/config.cmake /infra/config.cmake
COPY infra/install_tvm.sh /infra/install_tvm.sh
RUN bash /infra/install_tvm.sh

# copy the dependencies file to the working directory and install them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Environment variables
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/vta/python:${PYTHONPATH}
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Copy the source files to the working directory
COPY . .

# Run the container
CMD ["/bin/bash"]
