# Make file for the repository
# Comment: Any sane person would use CMake or similar for building
#          However, occationally files can be scattered all around the system
#          due to the setup of the GPUs
#          Hence, Makefile is chosen for this repository
#		   The only advantage is that the build commands becomes very
#          transparent as each command must be typed manually
#
# See https://makefiletutorial.com for a Makefile tutorial
# See https://developer.nvidia.com/blog/building-cuda-applications-cmake/ and
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html for a
# tutorial on CMake and Cuda

# The include directive tells make to suspend reading the current makefile and
# read one or more other makefiles before continuing
include common.mk

# Declaration of targets which doesn't output a file
.PHONY: all clean

# For all, we will simply call the sub-directories
# The first $ in $$ is an escape charater.
# So `$$dir` will expand to `$dir` instead of the variable named `dir`
# NOTE: When you call $(MAKE) -C you are creating a new process
# NOTE: The order does matter: We need dynamic_memory.o as a dependency.
#       This file is created in utils, and consumed as a dependency in for
#       example route.c, hence this needs to be processed first
COMPONENTS := snippets \
			  problem_sets
 all:
	for dir in $(COMPONENTS); do \
		$(MAKE) -C $$dir; \
	done

clean:
	rm -rf $(BUILD_DIR)
