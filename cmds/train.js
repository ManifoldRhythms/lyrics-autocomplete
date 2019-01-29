exports.command = 'train';
exports.description = '';
exports.builder = function(yargs) {
  return yargs
    .option('tpu_name', {
      default: process.env['TPU_NAME'],
      type: 'string',
      description:
        'The Cloud TPU to use for training. This should be the name used when creating the Cloud TPU. ' +
        "To find out the name of TPU, either use command 'gcloud compute tpus list --zone=<zone-name>', or " +
        "use 'ctpu status --details' if you have created Cloud TPU using 'ctpu up'."
    })
    .option('epochs', {
      default: 1,
      type: 'number',
      description: 'Total number of training and evaluation loops.'
    })
    .option('iterations', {
      default: 500,
      type: 'number',
      description: 'Number of iterations per TPU training loop.'
    })
    .option('batch_size', {
      default: 128,
      type: 'number',
      description: 'This is the global batch size and not the per-shard batch.'
    })
    .option('train_steps', {
      default: 2055,
      type: 'number',
      description: 'Total number of training steps.'
    })
    .option('eval_steps', {
      default: 4,
      type: 'number',
      description:
        'Total number of evaluation steps. If `0`, evaluation after training is skipped.'
    })
    .option('learning_rate', {
      default: 0.1,
      type: 'number',
      description: 'Learning rate'
    })
    .option('seq_len', {
      default: 30,
      type: 'number',
      description: 'Number of characters to split the dataset into.'
    });
};
exports.handler = async function(argv) {
  const { runCommand } = require('./utils');
  const { stdout, stderr } = await runCommand('python3', [
    'tf/train',
    `--tpu_name "${argv.tpu_name}"`,
    `--epochs ${argv.epochs}`,
    `--batch_size ${argv.batch_size}`,
    `--train_steps ${argv.train_steps}`,
    `--eval_steps ${argv.eval_steps}`,
    `--iterations ${argv.iterations}`,
    `--learning_rate ${argv.learning_rate}`,
    `--seq_len ${argv.seq_len}`,
    `--profile_every_n_steps 0`
  ]);
  console.log(stdout);
  if (stderr) {
    console.error(stderr);
  }
};
