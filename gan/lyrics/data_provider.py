import numpy as np
import tensorflow as tf

LYRICS_TXT = 'gs://mr-lyrics-autocomplete-data/lyrics.txt'

# tf.logging.set_verbosity(tf.logging.INFO)

def transform(txt, pad_to=None):
  # drop any non-ascii characters
  output = np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)
  if pad_to is not None:
    output = output[:pad_to]
    output = np.concatenate([
        np.zeros([pad_to - len(txt)], dtype=np.int32),
        output,
    ])
  return output

def training_generator(seq_len=100, batch_size=1024):
  """A generator yields (source, target) arrays for training."""
  with tf.gfile.GFile(LYRICS_TXT, 'r') as f:
    txt = f.read()

  # tf.logging.info('Input text [%d] %s', len(txt), txt[:50])
  source = transform(txt)
  while True:
    offsets = np.random.randint(0, len(source) - seq_len, batch_size)

    # Our model uses sparse crossentropy loss, but Keras requires labels
    # to have the same rank as the input logits.  We add an empty final
    # dimension to account for this.
    yield (
        np.stack([source[idx:idx + seq_len] for idx in offsets]),
        np.expand_dims(
            np.stack([source[idx + 1:idx + seq_len + 1] for idx in offsets]),
            -1),
    )
