#!/usr/bin/env bash

# Check if at least two arguments are provided (1 for directory, 1 for host)
if [ $# -lt 2 ]; then
    echo "Error: Usage: $0 <target_directory> <host1> [host2 ...]"
    exit 1
fi

# The first argument is the target directory
TARGET_DIR="$1"

# Check if the directory path is given
if [ -z "$TARGET_DIR" ]; then
    echo "Error: Target directory path not specified."
    exit 1
fi

# Skip the first argument and iterate over the rest (hosts)
shift
for HOST in "$@"; do
    echo "Deleting $TARGET_DIR on $HOST..."
    ssh "$HOST" "rm -rf $TARGET_DIR" &
done

echo "Operation completed."
