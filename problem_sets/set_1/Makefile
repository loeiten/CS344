include ../../common.mk

SET_1_SRC_DIR := $(ROOT_DIR)problem_sets/set_1/src
SET_1_INCL_DIR := $(ROOT_DIR)problem_sets/set_1/include

OPENCV_LD_LIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui

.PHONY: all clean

all: $(SET_1_EXEC_DIR)/problem_set_1

SET_1_HEADER_FILES := $(wildcard $(SET_1_INCL_DIR)/*.hpp)

SET_1_OBJ_FILES := $(SET_1_OBJ_DIR)/main.o \
	$(SET_1_OBJ_DIR)/student_func.o \
	$(SET_1_OBJ_DIR)/compare.o \
	$(SET_1_OBJ_DIR)/reference_calc.o

# FIXME: Copy the data files
$(SET_1_EXEC_DIR)/problem_set_1: $(SET_1_OBJ_FILES)
	$(call explain_build,Building and linking the binary file: $@)
	mkdir -p $(SET_1_EXEC_DIR)
	$(NVCC) $(CUDA_INCL) $(CUDA_LD_LIB) $(OPENCV_LD_LIB) $(OPENCV_LD_LIBS) $^ -o $@

# NOTE: The dependencies are explicitly included
#		Any change in these would result in a different .o file
$(SET_1_OBJ_DIR)/main.o: $(SET_1_SRC_DIR)/main.cpp $(SET_1_HEADER_FILES)
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_1_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CXX_SANITIZERS) $(CUDA_INCL) $(OPENCV_INCL) -c $< -o $@

$(SET_1_OBJ_DIR)/student_func.o: $(SET_1_SRC_DIR)/student_func.cu $(SET_1_INCL_DIR)/utils.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_1_OBJ_DIR)
	$(NVCC) $(COMMON_FLAGS) $(CUDA_INCL) --compiler-options "$(COMMON_LINT_FLAGS)" -c $< -o $@

$(SET_1_OBJ_DIR)/pre_post_process.o: $(SET_1_SRC_DIR)/pre_post_process.cpp $(SET_1_INCL_DIR)/pre_post_process.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_1_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CXX_SANITIZERS) $(OPENCV_INCL) -c $< -o $@

$(SET_1_OBJ_DIR)/compare.o: $(SET_1_SRC_DIR)/compare.cpp $(SET_1_INCL_DIR)/compare.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_1_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CXX_SANITIZERS) $(OPENCV_INCL) -c $< -o $@

$(SET_1_OBJ_DIR)/reference_calc.o: $(SET_1_SRC_DIR)/reference_calc.cpp $(SET_1_INCL_DIR)/reference_calc.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_1_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CXX_SANITIZERS) $(OPENCV_INCL) $(CUDA_INCL) -c $< -o $@

clean:
	rm -rf $(SET_1_OBJ_DIR) $(SET_1_EXEC_DIR)
