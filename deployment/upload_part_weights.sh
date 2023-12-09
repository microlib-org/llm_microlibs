#!/bin/bash

# Check if three arguments are provided
if [ $# -ne 3 ]; then
    echo "Error: Usage: $0 <source_directory> <destination_directory> <spec>"
    exit 1
fi

# Assign arguments to variables
SOURCE_DIR="$1"
DEST_DIR="$2"
SPEC="$3"

# Define the Python module
PYTHON_MODULE="llm_sepweight.filenames"

# Execute the Python command and loop over its output
for i in $(python -m $PYTHON_MODULE "$SPEC"); do
    # Perform the rsync operation
    rsync -avz --progress "$SOURCE_DIR/$i" "$DEST_DIR/$i"

    # Optional: Check if rsync was successful
    if [ $? -ne 0 ]; then
        echo "rsync failed for $i"
        # Decide how to handle the failure: exit, continue, etc.
    fi
done

echo "All rsync operations completed."
