
# C flag options
CC := clang
CFLAGS := -g -O0 -std=gnu17
C_LINT_FLAGS := -Wall -Wextra -Wpedantic -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Werror
C_SANITIZERS := -fsanitize=address -fsanitize=undefined

# CUDA specifics
CUDA_VERSION ?= 12.1
CUDA_ROOT_DIR ?= /usr/local/cuda-$(CUDA_VERSION)
CUDA_INC_DIR ?= -I$(CUDA_ROOT_DIR)/include
CUDA_LIB_DIR ?= -I$(CUDA_ROOT_DIR)/lib64

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
