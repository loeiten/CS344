include ../common.mk

.PHONY: all clean

SNIPPETS_SRC_FILES := $(wildcard $(ROOT_DIR)snippets/*.cu)
SNIPPETS_SRC := $(word 1, $(dir $(SNIPPETS_SRC_FILES)))
# Remove trailing slash
SNIPPETS_SRC := $(patsubst %/,%,$(SNIPPETS_SRC))
# NOTE:
# Substitution references
# $(var:pattern=replacement)
# are equivalent to
# $(patsubst %suffix,%replacement,$(var))
# Here we cannot use the former, and must use the latter
SNIPPETS_OBJ_FILES := $(patsubst $(SNIPPETS_SRC)%.cu,$(BUILD_OBJ_DIR)/snippets%.o,$(SNIPPETS_SRC_FILES))
SNIPPETS_EXEC_FILES := $(patsubst $(SNIPPETS_SRC)%.cu,$(EXEC_DIR)/snippets%,$(SNIPPETS_SRC_FILES))

# NOTE:
# Here we assume that all .cu file will result in an executable
all: $(SNIPPETS_EXEC_FILES)
# If we do not specify these as secondary, make will remove them as it considers
# them as intermediate
.SECONDARY: $(SNIPPETS_OBJ_FILES)

CUDA_LIBS := -lcudadevrt -lcudart_static

# Build the executables, one by one
$(EXEC_DIR)/snippets% : $(BUILD_OBJ_DIR)/snippets%.o
	$(call explain_build,Building and linking the binary: $@)
	mkdir -p $(SNIPPETS_EXEC_DIR)
	$(NVCC) $(CUDA_INCL) $(CUDA_LD_LIB) $(CUDA_LIBS) $(NVCC_LINT_FLAGS) $< -o $@

# Build the object file, one by one
# NOTE: We include the CUDA_INCL as this includes the system headers and
#       therefore suppress checking for errors in those files
$(BUILD_OBJ_DIR)/snippets%.o : $(SNIPPETS_SRC)%.cu
	$(call explain_build,Building the object file: $@)
	mkdir -p $(SNIPPETS_OBJ_DIR)
	$(NVCC) $(CUDA_INCL) $(NVCC_LINT_FLAGS) -c $< -o $@

clean:
	rm -rf $(SNIPPETS_OBJ_DIR) $(SNIPPETS_EXEC_DIR)
