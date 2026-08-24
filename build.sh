#!/bin/bash

# docker build -t hemidentification .

if [ ${#} -eq 1 ] ; then
    ver=${1}
    ver_string="--build-arg BUILD_VERSION=${ver}"
fi

docker buildx build \
    ${ver_string}                                      \
    -t hemidentification:${ver}                        \
    --cache-from type=local,src=docker_cache    \
    --cache-to type=local,dest=docker_cache     \
    .