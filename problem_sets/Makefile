.PHONY: all clean

COMPONENTS := set_1
all:
	for dir in $(COMPONENTS); do \
		$(MAKE) -C $$dir; \
	done

clean:
	rm -rf $(PROBLEM_SETS_OBJ_DIR) $(PROBLEM_SETS_EXEC_DIR)
