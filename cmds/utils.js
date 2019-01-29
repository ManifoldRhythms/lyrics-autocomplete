const util = require('util');
const child_process = require('child_process');
const execFile = util.promisify(child_process.execFile);

/**
 *
 * @param {string} command
 * @param {string[]} args
 * @param {({ encoding?: string | null } & child_process.ExecFileOptions) | undefined | null} options
 */
function runCommand(command, args, options) {
  return execFile(command, args, options);
}

module.exports = {
  runCommand
};
