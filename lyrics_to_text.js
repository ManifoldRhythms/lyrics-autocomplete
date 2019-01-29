const path = require('path');
const fs = require('fs');
const glob = require('glob');

const BASE_LYRICS_BATCH_DIR = path.resolve('./log/');
const BASE_LYRICS_TEXT_DIR = path.resolve('./lyrics/');

const files = glob.sync(
  path.join(BASE_LYRICS_BATCH_DIR, 'lyrics_batch_page*.json')
);

for (const filename of files) {
  /**
   * @type {{lyrics_id:number|string,lyrics_body:string}[]} data
   */
  const data = JSON.parse(fs.readFileSync(filename));

  console.log('loading batch', filename, `- ${data.length} records`);

  for (const d of data) {
    const outFilename = path.join(
      BASE_LYRICS_TEXT_DIR,
      `lyrics_${d.lyrics_id}.txt`
    );
    console.log('writing', outFilename);

    fs.writeFileSync(
      outFilename,
      d.lyrics_body.replace(
        '...\n\n******* This Lyrics is NOT for Commercial use *******',
        ''
      )
    );
  }
}
