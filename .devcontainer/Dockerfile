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
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt install -yqq\
    llvm-14-dev \
    libclang-14-dev \
    clang-14 \
    cppcheck \
    clang-format \
    clang-tidy \
    clangd \
    shellcheck \
    cmake \
    git \
    wget \
    pip \
    vim \
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

# Manually install IWYU in order to match the clang version (and to have fix_includes.py)
RUN git clone https://github.com/include-what-you-use/include-what-you-use.git && \
    cd ../include-what-you-use/ && \
    git checkout clang_14 && \
    mkdir build && \
    cd build && \
    cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/usr/lib/llvm-14 -DCMAKE_INSTALL_PREFIX:PATH=/usr .. && \
    cmake --build . --target install --config Release

# Update the alternatives
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# Remove unneeded files
RUN rm -rf \
    include-what-you-use \
    cuda_12.3.2_545.23.08_linux.run

# Set environment variables
# CUDA
ENV NVCC_PREPEND_FLAGS="-ccbin clang++"
ENV CUDA_VERSION=12.3

# OpenCV
# Needed for clang-tidy as one cannot add arguments after the file in pre-commit
ENV CPATH=/usr/local/cuda-12.3/include:usr/include/opencv4
ENV OPENCV_ROOT_DIR=
# Needed so that -Wpedantic don't emit errors about 3rd party libs
ENV OPENCV_INCL="-isystem /usr/include/opencv4"
# It appears that the libraries are placed in the normal search path for lib
# when installing through apt
ENV OPENCV_LD_LIB=
