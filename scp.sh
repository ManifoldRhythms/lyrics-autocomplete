#!/bin/bash

GCP_PROJECT_ID=mr-lyrics-autocomplete
REMOTE_USER=jeremy
REMOTE_INSTANCE_NAME=jeremy
REMOTE_INSTANCE_ZONE=us-central1-b
LOCAL_BASE_DIR=$(realpath ~/manifold_rhythms/lyrics-autocomplete)
REMOTE_BASE_DIR=/home/$REMOTE_USER/lyrics-autocomplete

gcloud compute scp $2 $LOCAL_BASE_DIR/$1 \
$REMOTE_USER@$REMOTE_INSTANCE_NAME:$REMOTE_BASE_DIR/$1 \
--zone $REMOTE_INSTANCE_ZONE \
--project $GCP_PROJECT_ID

# gcloud compute scp --recurse $LOCAL_BASE_DIR/[!.]* \
# $REMOTE_USER@$REMOTE_INSTANCE_NAME:$REMOTE_BASE_DIR \
# --scp-flag="-r" \
# --zone $REMOTE_INSTANCE_ZONE \
# --project $GCP_PROJECT_ID
