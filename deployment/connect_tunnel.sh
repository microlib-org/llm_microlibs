#!/bin/bash

# Check if the correct number of arguments is provided
if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <source_user@source_host:source_port> <target_host:target_port>"
    exit 1
fi

# Extracting source and target details
SOURCE_DETAILS="$1"
TARGET_DETAILS="$2"

SOURCE_USER="${SOURCE_DETAILS%@*}"
SOURCE_HOSTPORT="${SOURCE_DETAILS#*@}"
SOURCE_HOST="${SOURCE_HOSTPORT%:*}"
SOURCE_PORT="${SOURCE_HOSTPORT#*:}"

TARGET_HOST="${TARGET_DETAILS%:*}"
TARGET_PORT="${TARGET_DETAILS#*:}"

# Function to check if a given variable is a number
is_number() {
    re='^[0-9]+$'
    if ! [[ $1 =~ $re ]]; then
        echo "Error: $2 is not a number." >&2
        exit 1
    fi
}

# Check if provided ports are valid numbers
is_number "$SOURCE_PORT" "Source port"
is_number "$TARGET_PORT" "Target port"

# SSH and autossh commands
SSH_CMD="ssh -L $SOURCE_PORT:$TARGET_HOST:$TARGET_PORT $SOURCE_USER@$SOURCE_HOST"
AUTOSH_CMD="autossh -M 0 -N -L $SOURCE_PORT:$TARGET_HOST:$TARGET_PORT $SOURCE_USER@$SOURCE_HOST"

# Perform SSH connection check
echo "Checking SSH connection..."
$SSH_CMD -o BatchMode=yes -o ConnectTimeout=5 -q exit
if [[ "$?" -ne 0 ]]; then
    echo "SSH connection check failed. Exiting."
    exit 1
fi

# Initialize autossh connection
echo "Initializing autossh tunnel..."
$AUTOSH_CMD
if [[ "$?" -eq 0 ]]; then
    echo "Autossh tunnel established successfully."
else
    echo "Failed to establish autossh tunnel. Exiting."
    exit 1
fi
