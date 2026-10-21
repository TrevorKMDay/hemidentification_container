#!/bin/bash

# docker run                              \
#     -v .:/data                          \
#     hemidentification:latest            \
#         test                            \
#             /data/wab.pickle \
#             /data/hemiconnectome.pickle \
#             /data/wab_fit.csv

python bin/fingerprinting_manager.py    \
    -f strict                          \
    test                                \
        hemi.age_model.pickle \
        hemiconnectome.pickle \
        hemi.age_fit.csv