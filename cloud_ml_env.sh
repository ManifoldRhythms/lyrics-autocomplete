#!/bin/bash

JOB_NAME=tpu_1
STAGING_BUCKET=gs://mr-lyrics-autocomplete-data/
REGION=us-central1
DATA_DIR=gs://mr-lyrics-autocomplete-data/lyrics.txt
OUTPUT_PATH=gs://mr-lyrics-autocomplete-data/model/
# MODEL_DIR=$(realpath ./output)
MODEL_DIR=$OUTPUT_PATH
TRAIN_DATA=$DATA_DIR

# gcloud ml-engine jobs submit training $JOB_NAME \
#     --job-dir $OUTPUT_PATH \
#     --runtime-version 1.10 \
#     --module-name trainer.task \
#     --package-path trainer/ \
#     --region $REGION \
#     -- \
#     --train-files $TRAIN_DATA \
#     --eval-files $EVAL_DATA \
#     --train-steps 1000 \
#     --eval-steps 100 \
#     --verbosity DEBUG

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --train-steps 1000 \
    --eval-steps 100 \
    --verbosity DEBUG
