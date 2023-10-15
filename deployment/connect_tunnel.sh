#!/bin/bash

# Assign command line arguments to variables
INPUT_A=$1  # [user@]host for Host A
INPUT_B=$2  # [user@]host for Host B
PORT_A=$3   # Local port on Host A
PORT_B=$4   # Target port on Host B

# Parse user and host
parse_user_host() {
    local input=$1
    local user=${input%@*}
    local host=${input#*@}

    # If user is empty, use the current user
    [ -z "$user" ] && user=$USER

    echo "$user $host"
}

# Split user and host for A and B
USER_HOST_A=($(parse_user_host $INPUT_A))
USER_HOST_B=($(parse_user_host $INPUT_B))

USER_A=${USER_HOST_A[0]}
HOST_A=${USER_HOST_A[1]}
USER_B=${USER_HOST_B[0]}
HOST_B=${USER_HOST_B[1]}

# Check for input errors
if [[ -z $HOST_A ]] || [[ -z $HOST_B ]] || [[ -z $PORT_A ]] || [[ -z $PORT_B ]]; then
    echo "Usage: $0 [user_a@]host_a [user_b@]host_b port_a port_b"
    exit 1
fi

# Connect to Host A and from there, use autossh to set up the tunnel from Host A to Host B
ssh -t "$USER_A@$HOST_A" "autossh -M 0 -f -N -L $PORT_A:localhost:$PORT_B $USER_B@$HOST_B; sleep 10"
