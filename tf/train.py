import tensorflow as tf
import common
import lstm_model as model
import tpu_profiler_hook
import re

def train(argv):
    del argv  # Unused

    # Use all 8 cores for training
    estimator = model._make_estimator(num_shards=8, use_tpu=True, tpu_grpc_url=tpu_grpc_url)

    DATA_LEN = 7893287
    steps_per_epoch = DATA_LEN // (common.BATCHSIZE * common.SEQLEN)
    current_step = estimator.get_variable_value('global_step')

    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.',
                    steps_per_epoch * common.EPOCHS,
                    (steps_per_epoch * common.EPOCHS) / steps_per_epoch,
                    current_step)
    # hooks = []

    # if common.FLAGS.profile_every_n_steps > 0:
    #     hooks.append(
    #         tpu_profiler_hook.TPUProfilerHook(
    #             save_steps=common.FLAGS.profile_every_n_steps,
    #             output_dir=common.MODEL_LOG_DIR, tpu=common.FLAGS.tpu_name)
    #         )

    # for _ in range(common.EPOCHS):
    estimator.train(
        input_fn=model.input_fn,
        # steps=model.MAX_STEPS,
        steps=steps_per_epoch * common.EPOCHS,
        # hooks=hooks,
    )
    estimator.evaluate(input_fn=model.input_fn, steps=common.EVAL_STEPS)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=[common.TPU_WORKER])
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    tf.reset_default_graph()
    tf.set_random_seed(0)
    tf.app.run(train)