#!/bin/bash

# docker build -t hemidentification .

docker buildx build \
    -t hemidentification \
    --cache-from type=local,src=docker_cache \
    --cache-to type=local,dest=docker_cache \
    .