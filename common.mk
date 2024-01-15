
# Common options
COMMON_FLAGS := -g -O0 -std=c++17
COMMON_LINT_FLAGS := -Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Werror

# CXX flag options
CXX := clang++
# NOTE: CUDA is using `compute-sanitizer` instead of UBSAN and ASAN
# See
# https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/
# for details
CXX_SANITIZERS := -fsanitize=address -fsanitize=undefined
# -Wpedandtic throws errors like
#  error: style of line directive is a GCC extension
# See
# https://github.com/NVIDIA/cub/issues/228#issuecomment-741890136
# https://stackoverflow.com/questions/31000996/warning-when-compiling-cu-with-wpedantic-style-of-line-directive-is-a-gcc-ex
CXX_LINT_FLAGS := $(COMMON_LINT_FLAGS) -Wpedandtic

# CUDA specifics
CUDA_VERSION ?= 12.1
CUDA_ROOT_DIR ?= /usr/local/cuda-$(CUDA_VERSION)
# We do not use -I to declare that these are system headers
CUDA_INC_DIR ?= -isystem $(CUDA_ROOT_DIR)/include
CUDA_LIB_DIR ?= -L$(CUDA_ROOT_DIR)/lib64

NVCC := $(CUDA_ROOT_DIR)/bin/nvcc

# Setting up the paths
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(dir $(MKFILE_PATH))
BASE_DIR := $(abspath $(CUR_DIR)/..)
BUILD_DIR := $(abspath $(BASE_DIR)/build)
EXEC_DIR := $(abspath $(BUILD_DIR)/bin)
BUILD_OBJ_DIR := $(abspath $(BUILD_DIR)/obj_files)

# Define helper functions
define explain_build
	echo "\033[92m${1}\033[0m"
endef
define run_test
	echo "\033[94mRunning ${1}: ${2}...\033[0m"
	${2}
	echo "\033[94m...done\033[0m"
endef
