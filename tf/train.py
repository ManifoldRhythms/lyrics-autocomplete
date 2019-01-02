import tensorflow as tf
from common import TPU_WORKER
import lstm_model as model

EPOCHS=10

def train():
    # Use all 8 cores for training
    estimator = model._make_estimator(num_shards=8, use_tpu=True, tpu_grpc_url=tpu_grpc_url)

    for _ in range(EPOCHS):
        estimator.train(
            input_fn=model.input_fn,
            steps=model.MAX_STEPS,
        )
        estimator.evaluate(input_fn=model.input_fn, steps=100)

tf.logging.set_verbosity(tf.logging.INFO)
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[TPU_WORKER])
tpu_grpc_url = tpu_cluster_resolver.get_master()

tf.reset_default_graph()
tf.set_random_seed(0)

train()
