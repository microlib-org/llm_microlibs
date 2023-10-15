#!/bin/bash

# Assign command line arguments to variables
USER_A=$1  # User for Host A
HOST_A=$2  # Host A address
USER_B=$3  # User for Host B
HOST_B=$4  # Host B address
PORT_A=$5  # Local port on Host A
PORT_B=$6  # Target port on Host B

# Check for input errors
if [[ -z $USER_A ]] || [[ -z $HOST_A ]] || [[ -z $USER_B ]] || [[ -z $HOST_B ]] || [[ -z $PORT_A ]] || [[ -z $PORT_B ]]; then
    echo "Usage: $0 <user_a> <host_a> <user_b> <host_b> <port_a> <port_b>"
    exit 1
fi

# Connect to Host A and from there, use autossh to set up the tunnel from Host A to Host B
ssh -t "$USER_A@$HOST_A" "autossh -M 0 -f -N -L $PORT_A:localhost:$PORT_B $USER_B@$HOST_B"
