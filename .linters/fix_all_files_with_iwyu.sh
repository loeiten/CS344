#!/bin/bash

# Get the directory of the script
# $(dirname "$0") returns the directory of this script
# cd changes the directory to this directory
# If everything is successful pwd will "print working directory"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# We then obtain the directory which this directory is located in
PARENT_DIR="$(realpath "$(dirname "$SCRIPT_DIR")")"

# Find all .c and .h files recursively in the parent directory
# -type f - finds all file types
# -o is the logical or
# | will pipe the result to while
# read will read lines of input
# -r disables backslash escaping
# found_file is the where the result of the read is stored
find "$PARENT_DIR" -type f \( -name "*.cpp" -o -name "*.hpp" \) | while read -r found_file; do
    # Run my_command with the file as an argument
    include-what-you-use \
    -Xiwyu --mapping_file=iwyu.imp \
    -Xiwyu --update_comments \
    -I/usr/local/cuda-12.3/include \
    -I/usr/include/opencv4 \
    -std=c++17 \
    "$found_file" \
    > /tmp/iwyu.out 2>&1
    fix_includes.py \
    --comments \
    --update_comments \
    --nosafe_headers \
    < /tmp/iwyu.out
done
