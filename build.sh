#!/bin/bash

# docker build -t hemidentification .

if [ ${#} -eq 1 ] ; then
    ver=${1}
    ver_string="--build-arg BUILD_VERSION=${ver}"
else
    echo "Supply version string to embed"
    exit 1
fi

sed -I.bak "s/VERSION=.*/VERSION=\"${ver}\"/" bin/fingerprinting_manager.py

sed -I.bak "s/BUILD_DATE=.*/BUILD_DATE=\"$(date)\"/" \
        bin/fingerprinting_manager.py

echo
echo -n "Wrote version (${ver}) and date ($(date)) to "
echo    "fingerprinting_manager.py, starting build:"
echo

# shellcheck disable=SC2086
docker buildx build \
    ${ver_string}                                      \
    -t "hemidentification:${ver}"                        \
    -t "hemidentification:latest"                        \
    --cache-from type=local,src=docker_cache    \
    --cache-to type=local,dest=docker_cache     \
    .
