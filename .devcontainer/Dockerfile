# Start from base
# The platform is needed on Mac in order to avoid
# rosetta error: failed to open elf at /lib64/ld-linux-x86-64.so.2
FROM --platform=linux/amd64 ubuntu:22.04

# Intall apt dependencies
RUN apt -yqq update
RUN apt -yqq upgrade

# These two are needed so that tzdata can be installed
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install normal dependencies
# NOTE: It would be nice to use clang, but clang doesn't play too well with the
#       current CUDA version, giving errors like
#       /usr/bin/../lib/gcc/aarch64-linux-gnu/11/../../../../include/c++/11/ext/numeric_traits.h(70): error: qualified name is not allowed
#       Due to c++17 features
# NOTE: It would be nice to use the build-in gcc, but this also got the error
#       /usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with '...'
#       A workaround can be found at
#       https://github.com/NVlabs/instant-ngp/issues/119#issuecomment-1698809070
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt install -yqq\
    llvm \
    clang \
    cppcheck \
    iwyu \
    clang-format \
    clang-tidy \
    clangd \
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
# NOTE: Do NOT install nvidia-cuda-toolkit through apt as it gives a nightmare
#       when compiling with the underlying clang/gcc
RUN apt install -yqq \
    libopencv-dev

# Install CUDA from the runner
RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
RUN sh cuda_12.3.2_545.23.08_linux.run --toolkit --silent --override

# Set environment variables
# CUDA
ENV NVCC_PREPEND_FLAGS="-ccbin clang++"
ENV CUDA_VERSION=12.3

# OpenCV
# Needed for clang-tidy
ENV CPATH=/usr/include/opencv4
ENV OPENCV_ROOT_DIR=
# Needed so that -Wpedantic don't emit errors about 3rd party libs
ENV OPENCV_INCL="-isystem /usr/include/opencv4"
# It appears that the libraries are placed in the normal search path for lib
# when installing through apt
ENV OPENCV_LD_LIB=
