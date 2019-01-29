exports.command = 'list <command>';
exports.description = '';
exports.builder = function(yargs) {
  return yargs.commandDir('list_commands');
};
exports.handler = function(argv) {};
