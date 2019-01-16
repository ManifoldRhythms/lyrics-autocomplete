#!/bin/bash

# tensorboard --port 8080 \
# --logdir=gs://mr-lyrics-autocomplete-data/model/log/\
# ,gs://mr-lyrics-autocomplete-data/model/checkpoints/ \
# --alsologtostderr \
# --verbosity 0

tensorboard --port 8080 --logdir=gs://mr-lyrics-autocomplete-data/model/log

# capture_tpu_profile --tpu=$TPU_NAME --logdir=gs://mr-lyrics-autocomplete-data/model/log