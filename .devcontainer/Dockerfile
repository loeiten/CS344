## Build this
# docker build -t "cs344:1.0" .
## Mount dir and remove on exit
# docker run --rm -it -v $PWD:/current_pwd -w /current_pwd cs344:1.0

# Start from base
FROM ubuntu:latest

# Intall apt dependencies
RUN apt -yqq update
RUN apt -yqq upgrade

# Install normal dependencies
# These two are needed so that tzdata can be installed
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt install -yqq\
    llvm-12 \
    clang-12 \
    cppcheck \
    iwyu \
    clang-format \
    clang-tidy \
    clangd-12 \
    shellcheck \
    wget \
    pip \
    vim \
    git \
    ruby

# Install pre-commit
RUN pip install \
    pre-commit \
    cpplint

# Install project specific dependencies
RUN apt install -yqq \
    nvidia-cuda-toolkit \
    libopencv-dev

# Update the alternatives
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-12 100
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-12 100
RUN update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100

# Set environment variables
ENV CPATH=/usr/include/opencv4
ENV NVCC_PREPEND_FLAGS="-ccbin clang++"
ENV CUDA_ROOT_DIR=/usr
ENV CUDA_INCL=
ENV CUDA_LD_LIB=
