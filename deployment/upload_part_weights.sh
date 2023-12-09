#!/usr/bin/env bash

# Define variables
SOURCE_DIR="homebastion.local:/home/hvrigazov/llm-sepweights/falcon-7b"
DEST_DIR="/home/hvrigazov/llm-sepweights/falcon-7b"
SPEC="16-32 e"
PYTHON_MODULE="llm_sepweight.filenames"

# Execute the Python command and loop over its output
for i in $(python -m $PYTHON_MODULE "$PYTHON_ARGS"); do
    # Perform the rsync operation
    rsync -avz --progress "$SOURCE_DIR/$i" "$DEST_DIR/$i"

    # Optional: Check if rsync was successful
    if [ $? -ne 0 ]; then
        echo "rsync failed for $i"
        # Decide how to handle the failure: exit, continue, etc.
    fi
done

echo "All rsync operations completed."
