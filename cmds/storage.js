exports.command = 'storage <command>';
exports.description = '';
exports.builder = function(yargs) {
  return yargs.commandDir('storage_commands');
};
exports.handler = function(argv) {};
