# set base image
# FROM public.ecr.aws/docker/library/python:3.10

# Minimum docker image for demo purposes
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# WORKDIR
ENV APP_HOME=/app
RUN mkdir -p $APP_HOME

# Set the working directory in the container
WORKDIR $APP_HOME

# Install git
RUN apt-get update \
    && apt-get -y install git cmake \
    && apt-get clean

# COPY utils/apt-install-and-clear.sh /usr/local/bin/apt-install-and-clear

# Build TVM
COPY infra/install_tvm_gpu.sh /infra/install_tvm_gpu.sh
RUN bash /infra/install_tvm_gpu.sh

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
