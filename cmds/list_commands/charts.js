const path = require('path');
const fs = require('fs');
const helper = require(path.join(__dirname, './helper.js'));

exports.command = 'charts';
exports.desc = 'List of the top songs.';
exports.builder = function(yargs) {
  return yargs
    .option('output_file', {
      description:
        'Write results to file. Filename format: chart_list_page_[page].json',
      type: 'boolean',
      default: true
    })
    .option('output_directory', {
      alias: 'dir',
      description: 'Write results to directory.',
      type: 'string',
      default: '.',
      coerce: path.resolve
    })
    .option('page', {
      description: 'The page number for paginated results.',
      type: 'number',
      default: 1
    })
    .option('page_size', {
      description: 'The page size for paginated results. Range is 1 to 100.',
      type: 'number',
      default: 10
    })
    .option('has_lyrics', {
      description: 'When set, filter only contents with lyrics.',
      type: 'boolean',
      default: true
    });
};
exports.handler = async function(argv) {
  try {
    const res = await helper.listChart({
      page: argv.page,
      page_size: argv.page_size
    });

    const out = JSON.stringify(res, null, 2);

    if (argv.output_file) {
      const out_filename = path.join(
        argv.output_directory,
        `chart_list_page_${argv.page.toString().padStart(4, '0')}.json`
      );
      fs.writeFileSync(out_filename, out);
      console.log(`Wrote file ${out_filename}`);
    } else {
      console.log(JSON.stringify(res, null, 2));
    }
  } catch (err) {
    console.error(err);
  }
};
