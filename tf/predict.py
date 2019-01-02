import tensorflow as tf
from common import TPU_WORKER
import lstm_model as model

def _seed_input_fn(params):
  del params
  seed_txt = 'I love tensorflow'
  seed = model.transform(seed_txt)
  seed = tf.constant(seed.reshape([1, -1]), dtype=tf.int32)
  # Predict must return a Dataset, not a Tensor.
  return tf.data.Dataset.from_tensors({'source': seed})

def predict():
    # Use 1 core for prediction since we're only generating a single element batch
    estimator = model._make_estimator(num_shards=None, use_tpu=True, tpu_grpc_url=tpu_grpc_url)

    idx = next(estimator.predict(input_fn=_seed_input_fn))['predictions']
    print(''.join([chr(i) for i in idx]))

tf.logging.set_verbosity(tf.logging.INFO)
tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[TPU_WORKER])
tpu_grpc_url = tpu_cluster_resolver.get_master()

tf.reset_default_graph()
tf.set_random_seed(0)

predict()
