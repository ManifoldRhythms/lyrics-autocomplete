import tensorflow as tf
from tensorflow.contrib import tpu
import sys, os
import pprint
import re
import numpy as np
from random import randrange
import common

RANDOM_SEED = 42  # An arbitrary choice.
MAX_STEPS=common.MAX_STEPS
BATCHSIZE = common.BATCHSIZE
EMBEDDING_DIM = 1024
learning_rate = common.LEARNING_RATE

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def transform(txt):
    return np.asarray([ord(c) for c in txt], dtype=np.int32)

def input_fn(params):
    """Return a dataset of source and target sequences for training."""
    batch_size = params['batch_size']
    tf.logging.info('Batch size: {}'.format(batch_size))
    seq_len = params['seq_len']
    filename = params['source_filename']
    all_txt = ''

    with tf.gfile.GFile(filename, 'r') as f:
        tf.logging.info("Loading file {}".format(filename))
        txt = f.read()
        all_txt = all_txt + ''.join([x for x in txt if ord(x) < 128])

    # tf.logging.info('Sample text: {} \n\n(length={})'.format(all_txt[1000:1010], len(all_txt)))

    # randomly sample 10 characters from text dataset
    all_text_len = len(all_txt)
    sample_text_start = randrange(0, all_text_len-10)
    tf.logging.info('Sample text: {} \n\n(length={})'.format(all_txt[sample_text_start:sample_text_start+10], all_text_len))

    source = tf.constant(transform(all_txt), dtype=tf.int32)
    ds = tf.data.Dataset.from_tensors(source)
    ds = ds.repeat()
    ds = ds.apply(tf.data.experimental.enumerate_dataset())

    def _select_seq(offset, src):
        idx = tf.contrib.stateless.stateless_random_uniform(
            [1], seed=[RANDOM_SEED, offset], dtype=tf.float32)[0]

        max_start_offset = len(all_txt) - seq_len
        idx = tf.cast(idx * max_start_offset, tf.int32)

        return {
            'source': tf.reshape(src[idx:idx + seq_len], [seq_len]),
            'target': tf.reshape(src[idx + 1:idx + seq_len + 1], [seq_len])
        }

    ds = ds.map(_select_seq, num_parallel_calls=100)
    # ds = ds.shuffle(buffer_size=8*1024*1024)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(BATCHSIZE)
    return ds

# Construct a 2-layer LSTM
def _lstm(inputs, batch_size, initial_state=None):
    def _make_cell(layer_idx):
        with tf.variable_scope('lstm/%d' % layer_idx,):
            return tf.nn.rnn_cell.LSTMCell(
                num_units=EMBEDDING_DIM,
                state_is_tuple=True,
                reuse=tf.AUTO_REUSE,
            )

    cell = tf.nn.rnn_cell.MultiRNNCell([
        _make_cell(0), 
        _make_cell(1), 
        # _make_cell(2),
    ])
    if initial_state is None:
        initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.contrib.recurrent.functional_rnn(
        cell, inputs, initial_state=initial_state, use_tpu=True)
    return outputs, final_state

def lstm_model(seq, initial_state=None):
    with tf.variable_scope('lstm', 
                            initializer=tf.orthogonal_initializer,
                            reuse=tf.AUTO_REUSE):
        batch_size = seq.shape[0]
        seq_len = seq.shape[1]

        embedding_params = tf.get_variable(
            'char_embedding', 
            initializer=tf.orthogonal_initializer(seed=0),
            shape=(256, EMBEDDING_DIM), dtype=tf.float32)

        embedding = tf.nn.embedding_lookup(embedding_params, seq)

        lstm_output, lstm_state = _lstm(
            embedding, batch_size, initial_state=initial_state)

        # Apply a single dense layer to the output of our LSTM to predict
        # our final characters.  This looks awkward as we have to flatten
        # our input to 2 dimensions before applying the dense layer.
        flattened = tf.reshape(lstm_output, [-1, EMBEDDING_DIM])
        logits = tf.layers.dense(flattened, 256, name='logits',)
        logits = tf.reshape(logits, [-1, seq_len, 256])
        return logits, lstm_state

def train_fn(source, target):
    logits, lstm_state = lstm_model(source)
    batch_size = source.shape[0]

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if common.TPU_WORKER:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op,
    )

def eval_fn(source, target):
    logits, _ = lstm_model(source)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logits))

    def metric_fn(labels, logits):
        labels = tf.cast(labels, tf.int64)

        # accuracy = tf.metrics.accuracy(labels, logits)

        return {
            'recall@1': tf.metrics.recall_at_k(labels, logits, 1),
            'recall@5': tf.metrics.recall_at_k(labels, logits, 5),
            'precision@1': tf.metrics.precision_at_k(labels, logits, 1),
            'precision@5': tf.metrics.precision_at_k(labels, logits, 5),
            # 'accuracy': accuracy
        }

    eval_metrics = (metric_fn, [target, logits])
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL, 
        loss=loss, 
        eval_metrics=eval_metrics)

def predict_fn(source):
    # Seed the model with our initial array
    batch_size = source.shape[0]
    logits, lstm_state = lstm_model(source)

    def _body(i, state, preds):
        """Body of our prediction loop: predict the next character."""
        cur_preds = preds.read(i)
        next_logits, next_state = lstm_model(
            tf.cast(tf.expand_dims(cur_preds, -1), tf.int32), state)

        # pull out the last (and only) prediction.
        next_logits = next_logits[:, -1]
        next_pred = tf.multinomial(
            next_logits, num_samples=1, output_dtype=tf.int32)[:, 0]
        preds = preds.write(i + 1, next_pred)
        return (i + 1, next_state, preds)

    def _cond(i, state, preds):
        del state
        del preds

        # Loop until `predict_len - 1`: preds[0] is the initial state and we
        # write to `i + 1` on each iteration.
        return tf.less(i, predict_len - 1)

    next_pred = tf.multinomial(
        logits[:, -1], num_samples=1, output_dtype=tf.int32)[:, 0]

    i = tf.constant(0, dtype=tf.int32)

    predict_len = 500

    # compute predictions as [seq_len, batch_size] to simplify indexing/updates
    pred_var = tf.TensorArray(
        dtype=tf.int32,
        size=predict_len,
        dynamic_size=False,
        clear_after_read=False,
        element_shape=(batch_size,),
        name='prediction_accumulator',
    )

    pred_var = pred_var.write(0, next_pred)
    _, _, final_predictions = tf.while_loop(_cond, _body,
                                            [i, lstm_state, pred_var])

    # reshape back to [batch_size, predict_len] and cast to int32
    final_predictions = final_predictions.stack()
    final_predictions = tf.transpose(final_predictions, [1, 0])
    final_predictions = tf.reshape(final_predictions, (batch_size, predict_len))

    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT, 
        predictions={'predictions': final_predictions})

def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return train_fn(features['source'], features['target'])
    if mode == tf.estimator.ModeKeys.EVAL:
        return eval_fn(features['source'], features['target'])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return predict_fn(features['source'])

def _make_estimator(num_shards, use_tpu=True, tpu_grpc_url=None):
    config = tf.contrib.tpu.RunConfig(
        tf_random_seed=RANDOM_SEED,
        master=tpu_grpc_url,
        model_dir=common.MODEL_CHECKPOINTS_DIR,
        save_checkpoints_steps=2000,
        save_summary_steps=100,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=num_shards, iterations_per_loop=common.ITERATIONS_PER_LOOP))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn, config=config,
        train_batch_size=BATCHSIZE,
        eval_batch_size=BATCHSIZE,
        predict_batch_size=BATCHSIZE,
        params={'seq_len': common.SEQLEN, 'source_directory': common.TRAINING_DATA_DIR, 'source_filename': common.TRAINING_DATA_FILENAME},
    )
    return estimator
