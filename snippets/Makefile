include ../common.mk

.PHONY: all clean

SNIPPETS_SRC_FILES := $(wildcard $(CUR_DIR)snippets/*.cu)
SNIPPETS_SRC := $(word 1, $(dir $(SNIPPETS_SRC_FILES)))
# Remove trailing slash
SNIPPETS_SRC := $(patsubst %/,%,$(SNIPPETS_SRC))
# NOTE:
# Substitution references
# $(var:pattern=replacement)
# are equivalent to
# $(patsubst %suffix,%replacement,$(var))
# Here we cannot use the former, and must use the latter
SNIPPETS_OBJ_FILES := $(patsubst $(SNIPPETS_SRC)%.cu,$(BUILD_OBJ_DIR)%.o,$(SNIPPETS_SRC_FILES))
SNIPPETS_EXE_FILES := $(patsubst $(SNIPPETS_SRC)%.cu,$(EXEC_DIR)%,$(SNIPPETS_SRC_FILES))

# NOTE:
# Here we assume that all .cu file will result in an executable
all: $(SNIPPETS_EXE_FILES)

# Build the executables, one by one
$(SNIPPETS_EXE_FILES) : $(SNIPPETS_OBJ_FILES)
	$(call explain_build,Building and linking the binaries for snippets)
	$(NVCC) $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $< -o $@

# Build the object file, one by one
$(BUILD_OBJ_DIR)%.o : $(SNIPPETS_SRC)%.cu
	$(call explain_build,Building the object files for snippets)
	mkdir -p $(BUILD_OBJ_DIR)
	$(NVCC) -c $< -o $@

clean:
	rm -rf $(EXEC_DIR) $(BUILD_OBJ_DIR)
