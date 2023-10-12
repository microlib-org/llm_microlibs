#!/usr/bin/env sh

# Check if argument is not supplied
if [ -z "$1" ]; then
    echo "Usage: $0 <hostname>"
    echo "Please provide the hostname as an argument."
    exit 1
fi

NEXT_HOST=$1
ssh -L 61001:127.0.0.1:61000 "$NEXT_HOST"
