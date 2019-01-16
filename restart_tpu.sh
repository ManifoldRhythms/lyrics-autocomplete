#!/bin/bash

PROJECT_ID=$(gcloud config list --format="value(core.project)")
TPU_ZONE=$(gcloud config list --format="value(compute.zone)")
TPU_NAME=${1:-$TPU_NAME}

echo 'List TPU'
echo '
gcloud compute tpus list \
--filter="name:'$TPU_NAME'" \
--zone='$TPU_ZONE' \
--project='$PROJECT_ID'
'
gcloud compute tpus list \
--filter="name:$TPU_NAME" \
--zone=$TPU_ZONE \
--project=$PROJECT_ID

echo 'Stopping TPU'
echo '
gcloud compute tpus stop \
'$TPU_NAME' \
--zone='$TPU_ZONE' \
--project='$PROJECT_ID'
'
gcloud compute tpus stop \
$TPU_NAME \
--zone=$TPU_ZONE \
--project=$PROJECT_ID

echo 'Starting TPU'
echo '
gcloud compute tpus start \
'$TPU_NAME' \
--zone='$TPU_ZONE' \
--project='$PROJECT_ID'
'
gcloud compute tpus start \
$TPU_NAME \
--zone=$TPU_ZONE \
--project=$PROJECT_ID
