#!/usr/bin/env sh
NEXT_HOST=$1
ssh -L 61001:127.0.0.1:61000 "$NEXT_HOST"
