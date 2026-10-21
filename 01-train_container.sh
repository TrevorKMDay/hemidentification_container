#!/bin/bash

docker run                              \
    -v .:/data                          \
    hemidentification:latest            \
        -f compare                      \
        train                           \
            /data/hemiconnectome.pickle \
            /data/hemi.age_model.pickle      \
            hemi age_group