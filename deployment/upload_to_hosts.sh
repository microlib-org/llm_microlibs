#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <directory_path> <host1> [<host2> ... <hostN>]"
    exit 1
fi

# Get the directory path and hosts as arguments
DIR_PATH="$1"
shift
HOSTS=("$@")

# Check if the provided directory path exists
if [ ! -d "$DIR_PATH" ]; then
    echo "Error: The provided directory path does not exist."
    exit 1
fi

# Ensure directory path is not ended with /
DIR_PATH="${DIR_PATH%/}"
DIR_PATH=$(realpath "$DIR_PATH")

# Iterate over each host and use rsync to copy the directory
for HOST in "${HOSTS[@]}"; do
    echo "Syncing $DIR_PATH to $HOST:$DIR_PATH..."

    # Use rsync to copy the directory to the host
    # -a: archive mode, ensures that symbolic links, devices, attributes, permissions, ownerships, etc. are preserved
    # -v: verbose, to show the sync process in the terminal
    # -e: specify the shell to use for synchronization
    # --delete: delete extraneous files from destination dirs
    # --progress: show progress during transfer
    rsync -av -e ssh --progress "$DIR_PATH/" "$HOST":"$DIR_PATH" --rsync-path="mkdir -p $DIR_PATH && rsync"

    # Check the exit status of rsync and report accordingly
    if [ "$?" -eq 0 ]; then
        echo "Sync to $HOST completed successfully!"
    else
        echo "Error: rsync to $HOST failed. Check the network and try again."
    fi
done
