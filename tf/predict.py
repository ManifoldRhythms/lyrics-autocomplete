import tensorflow as tf
import common
import lstm_model as model

def _seed_input_fn(params):
  del params
  seed_txt = 'I am blind; the truth is screaming at me'
  seed = model.transform(seed_txt)
  seed = tf.constant(seed.reshape([1, -1]), dtype=tf.int32)
  # Predict must return a Dataset, not a Tensor.
  return tf.data.Dataset.from_tensors({'source': seed})

def predict(argv):
    del argv  # Unused

    # Use 1 core for prediction since we're only generating a single element batch
    estimator = model._make_estimator(num_shards=None, use_tpu=False, tpu_grpc_url=tpu_grpc_url)

    idx = next(estimator.predict(input_fn=_seed_input_fn))['predictions']
    print(''.join([chr(i) for i in idx]))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[common.TPU_WORKER])
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    tf.reset_default_graph()
    tf.set_random_seed(0)
    tf.app.run(predict)
