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
# NOTE: These should probably be set manually
CUDA_VERSION ?= 12.1
CUDA_ROOT_DIR ?= /usr/local/cuda-$(CUDA_VERSION)
# We do not use -I to declare that these are system headers
CUDA_INCL ?= -isystem $(CUDA_ROOT_DIR)/include
CUDA_LD_LIB ?= -L$(CUDA_ROOT_DIR)/lib64
# NOTE: For optimized code use -O3 togehter with -arch flags
#       See
#       https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
#       https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute
#       PTX is a low-level parallel-thread-execution virtual machine and ISA (Instruction Set Architecture)
#       SASS (source and assembly) is the low-level assembly language that compiles to binary microcode, which executes natively on NVIDIA GPU hardware.
NVCC := $(CUDA_ROOT_DIR)/bin/nvcc

# OpenCV specifics
OPENCV_ROOT_DIR ?= /usr
OPENCV_INCL := -isystem $(OPENCV_ROOT_DIR)/include
OPENCV_LD_LIB := -L$(OPENCV_ROOT_DIR)/lib64

# Setting up the paths
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(dir $(MKFILE_PATH))
BUILD_DIR := $(abspath $(ROOT_DIR)/build)
EXEC_DIR := $(abspath $(BUILD_DIR)/bin)
BUILD_OBJ_DIR := $(abspath $(BUILD_DIR)/obj_files)
# Snippets
SNIPPETS_OBJ_DIR := $(abspath $(BUILD_OBJ_DIR)/snippets)
SNIPPETS_EXEC_DIR := $(abspath $(EXEC_DIR)/snippets)
# Problem sets
PROBLEM_SETS_OBJ_DIR := $(abspath $(BUILD_OBJ_DIR)/problem_sets)
PROBLEM_SETS_EXEC_DIR := $(abspath $(EXEC_DIR)/problem_sets)
SET_1_OBJ_DIR := $(abspath $(PROBLEM_SETS_OBJ_DIR)/set_1)
SET_1_EXEC_DIR := $(abspath $(PROBLEM_SETS_EXEC_DIR)/set_1)


# Define helper functions
define explain_build
	echo "\033[92m${1}\033[0m"
endef
define run_test
	echo "\033[94mRunning ${1}: ${2}...\033[0m"
	${2}
	echo "\033[94m...done\033[0m"
endef
