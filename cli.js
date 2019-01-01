#!/usr/bin/env node

require('yargs')
  .demand(1)
  .commandDir('cmds')
  .wrap(120)
  .recommendCommands()
  .help()
  .strict().argv;
