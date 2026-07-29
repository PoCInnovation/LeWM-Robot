#!/bin/bash

ARCH=$1

if [[ -z "$ARCH" ]]; then
    echo "Usage: ./build.sh <arch>  (ada | blackwell)"
    exit 1
fi

if [[ "$ARCH" != "ada" && "$ARCH" != "blackwell" ]]; then
    echo "Error: arch must be 'ada' or 'blackwell', got '$ARCH'"
    exit 1
fi

docker build -t real-robot -f "docker/real/Dockerfile.${ARCH}" .