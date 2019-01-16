"""Evaluation Hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import common
import lstm_model as model


class EvalHook(tf.train.SessionRunHook):
  def __init__(self,
               estimator,
               save_steps=None):
              #  save_secs=None,):
    self._timer = tf.train.SecondOrStepTimer(every_steps=save_steps)
    self._running_process = None
    self._ran_first_step = False
    self.estimator = estimator

  def begin(self):
    self._global_step_tensor = tf.train.get_or_create_global_step()  # pylint: disable=protected-access

  def before_run(self, run_context):
    return tf.train.SessionRunArgs({"global_step": self._global_step_tensor})

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results["global_step"]
    if not self._ran_first_step:
      # Update the timer so that it does not activate until N steps or seconds
      # have passed.
      self._timer.update_last_triggered_step(stale_global_step)
      self._ran_first_step = True

    global_step = stale_global_step + 1
    if (stale_global_step > 1 and
        self._timer.should_trigger_for_step(stale_global_step)):
      global_step = run_context.session.run(self._global_step_tensor)
      self._timer.update_last_triggered_step(global_step)
      self.estimator.evaluate(input_fn=model.input_fn, steps=common.EVAL_STEPS)
