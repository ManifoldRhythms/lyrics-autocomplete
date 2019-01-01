import os

GCS_BUCKET_NAME = "gs://mr-lyrics-autocomplete-data/"
TPU_WORKER = os.environ['TPU_NAME']

MODEL_ROOT_DIR = os.path.join(GCS_BUCKET_NAME, "model/")
MODEL_LOG_DIR = os.path.join(MODEL_ROOT_DIR, "log/")
MODEL_CHECKPOINTS_DIR = MODEL_LOG_DIR # os.path.join(MODEL_ROOT_DIR, "checkpoints/")
MODEL_LOG_TRAIN_DIR = os.path.join(MODEL_LOG_DIR, "train/")
MODEL_LOG_EVAL_DIR = os.path.join(MODEL_LOG_DIR, "eval/")

TRAINING_DATA_DIR = os.path.join(GCS_BUCKET_NAME, "lyrics/")
TRAINING_DATA_FILENAME = os.path.join(GCS_BUCKET_NAME, "lyrics_data.txt")
