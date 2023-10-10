#!/bin/bash

# Check for the correct number of command-line arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_directory> <output_directory> <host1> <host2> ..."
    exit 1
fi

# Get the source directory, output directory, and shift the arguments
source_directory="$1"
output_directory="$2"
shift 2

# Check if the source directory exists
if [ ! -d "$source_directory" ]; then
    echo "Source directory '$source_directory' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through the list of hosts provided as arguments
for host in "$@"; do
    # Define the output directory on the local machine based on the host
    host_output_directory="${output_directory}/${host}"

    # Use scp to copy the source directory from the remote host to the local output directory
    scp -r "$host":"$source_directory" "$host_output_directory"

    # Check if scp was successful
    if [ $? -eq 0 ]; then
        echo "Successfully copied from $host:$source_directory to $host_output_directory"
    else
        echo "Failed to copy from $host:$source_directory to $host_output_directory"
    fi
done
