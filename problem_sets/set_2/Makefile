include ../../common.mk

SET_2_SRC_DIR := $(ROOT_DIR)problem_sets/set_2/src
SET_2_INCL_DIR := $(ROOT_DIR)problem_sets/set_2/include

OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core
CUDA_LIBS := -lcudadevrt -lcudart_static

.PHONY: all clean

all: $(SET_2_EXEC_DIR)/problem_set_2

SET_2_HEADER_FILES := $(wildcard $(SET_2_INCL_DIR)/*.hpp)

SET_2_OBJ_FILES := $(SET_2_OBJ_DIR)/main.o \
	$(SET_2_OBJ_DIR)/student_func.o \
	$(SET_2_OBJ_DIR)/image.o \
	$(SET_2_OBJ_DIR)/compare.o \
	$(SET_2_OBJ_DIR)/reference_calc.o

$(SET_2_EXEC_DIR)/problem_set_2: $(SET_2_OBJ_FILES)
	$(call explain_build,Building and linking the binary file: $@)
	mkdir -p $(SET_2_EXEC_DIR)
	cp -r $(SET_2_SRC_DIR)/../data $(SET_2_EXEC_DIR)
	$(NVCC) $(COMMON_FLAGS) $(CUDA_INCL) $(CUDA_LD_LIB) $(CUDA_LIBS) $(OPENCV_LD_LIB) $(OPENCV_LIBS) $(NVCC_LINT_FLAGS) $^ -o $@

# NOTE: The dependencies are explicitly included
#		Any change in these would result in a different .o file
$(SET_2_OBJ_DIR)/main.o: $(SET_2_SRC_DIR)/main.cpp $(SET_2_HEADER_FILES)
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_2_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CUDA_INCL) $(OPENCV_INCL) -c $< -o $@

$(SET_2_OBJ_DIR)/student_func.o: $(SET_2_SRC_DIR)/student_func.cu $(SET_2_INCL_DIR)/utils.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_2_OBJ_DIR)
	$(NVCC) $(COMMON_FLAGS) $(CUDA_INCL) $(NVCC_LINT_FLAGS) -c $< -o $@

$(SET_2_OBJ_DIR)/image.o: $(SET_2_SRC_DIR)/image.cpp $(SET_2_INCL_DIR)/image.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_2_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CUDA_INCL) $(OPENCV_INCL) -c $< -o $@

$(SET_2_OBJ_DIR)/compare.o: $(SET_2_SRC_DIR)/compare.cpp $(SET_2_INCL_DIR)/compare.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_2_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(CUDA_INCL) $(OPENCV_INCL) -c $< -o $@

$(SET_2_OBJ_DIR)/reference_calc.o: $(SET_2_SRC_DIR)/reference_calc.cpp $(SET_2_INCL_DIR)/reference_calc.hpp
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SET_2_OBJ_DIR)
	$(CXX) $(COMMON_FLAGS) $(CXX_LINT_FLAGS) $(OPENCV_INCL) $(CUDA_INCL) -c $< -o $@

clean:
	rm -rf $(SET_2_OBJ_DIR) $(SET_2_EXEC_DIR)
