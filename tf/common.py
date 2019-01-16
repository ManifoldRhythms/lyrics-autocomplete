import os
import tensorflow as tf

tf.flags.DEFINE_string(
    "tpu_name", default=os.getenv('TPU_NAME'),
    help="The Cloud TPU to use for training. This should be the name used when "
    "creating the Cloud TPU. To find out the name of TPU, either use command "
    "'gcloud compute tpus list --zone=<zone-name>', or use "
    "'ctpu status --details' if you have created Cloud TPU using 'ctpu up'.")

# Model specific parameters
tf.flags.DEFINE_string(
    "model_dir", default="gs://mr-lyrics-autocomplete-data/",
    help="This should be the path of GCS bucket which will be used as "
    "model_directory to export the checkpoints during training.")
tf.flags.DEFINE_integer(
    "epochs", default=1,
    help="Total number of training and evaluation loops.")
tf.flags.DEFINE_integer(
    "batch_size", default=128,
    help="This is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer(
    "train_steps", default=2055,
    help="Total number of training steps.")
tf.flags.DEFINE_integer(
    "eval_steps", default=4,
    help="Total number of evaluation steps. If `0`, evaluation "
    "after training is skipped.")
tf.flags.DEFINE_float(
    "learning_rate", default=0.1,
    help="Learning rate")
tf.flags.DEFINE_integer(
    "seq_len", default=30,
    help="Number of characters to split the dataset into.")

# TPU specific parameters.
tf.flags.DEFINE_bool(
    "use_tpu", default=True,
    help="True, if want to run the model on TPU. False, otherwise.")
tf.flags.DEFINE_integer(
    "iterations", default=500,
    help="Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer(
    'profile_every_n_steps', default=0,
    help=('Number of steps between collecting profiles if larger than 0'))

tf.flags.DEFINE_bool(
    "dry_run", default=True,
    help="")

FLAGS = tf.flags.FLAGS

GCS_BUCKET_NAME = FLAGS.model_dir
MODEL_ROOT_DIR = os.path.join(GCS_BUCKET_NAME, "model/")
MODEL_LOG_DIR = os.path.join(MODEL_ROOT_DIR, "log/")
MODEL_CHECKPOINTS_DIR = MODEL_LOG_DIR # os.path.join(MODEL_ROOT_DIR, "checkpoints/")
MODEL_LOG_TRAIN_DIR = os.path.join(MODEL_LOG_DIR, "train/")
MODEL_LOG_EVAL_DIR = os.path.join(MODEL_LOG_DIR, "eval/")
TRAINING_DATA_DIR = os.path.join(GCS_BUCKET_NAME, "lyrics/")
TRAINING_DATA_FILENAME = os.path.join(GCS_BUCKET_NAME, "lyrics.txt")

TPU_WORKER = FLAGS.tpu_name
EPOCHS=FLAGS.epochs
BATCHSIZE = FLAGS.batch_size
SEQLEN = FLAGS.seq_len

# Number of iterations per TPU training loop.
ITERATIONS_PER_LOOP=FLAGS.iterations

# Total number of training steps.
MAX_STEPS=FLAGS.train_steps

# Total number of evaluation steps.
EVAL_STEPS=FLAGS.eval_steps

# fixed learning rate
LEARNING_RATE = FLAGS.learning_rate
