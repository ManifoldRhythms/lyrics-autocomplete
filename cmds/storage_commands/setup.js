const path = require('path');
const fs = require('fs');
const tmp = require('tmp');
const rp = require('request-promise');
const request = require('request');
const { Storage } = require('@google-cloud/storage');
const LYRICS_FILE_URL =
  'https://storage.googleapis.com/mr-lyrics-autocomplete-data/lyrics.txt';

/**
 * Download url to a temporary file.
 * @param {string} url
 * @returns {Promise<tmp.SynchrounousResult>}
 */
function downloadFile(url) {
  return new Promise((resolve, reject) => {
    const tmpobj = tmp.fileSync();
    const fsStream = fs.createWriteStream(tmpobj.name);

    request
      .get(url)
      .pipe(fsStream)
      .on('error', reject)
      .on('close', () => {
        console.log('download complete', tmpobj.name);
        resolve(tmpobj);
      });
  });
}

/**
 *
 * @param {Storage} storage
 * @param {string} bucketName
 * @param {string} filename
 */
async function uploadFile(storage, bucketName, filename) {
  const [file] = await storage.bucket(bucketName).upload(filename, {
    gzip: true,
    metadata: {
      cacheControl: 'no-cache'
    },
    destination: 'lyrics_data.txt'
  });

  console.log(`${file.name} uploaded to ${file.bucket.name}.`);

  return file;
}

exports.command = 'setup';
exports.desc = 'Setup the Google Cloud Storage for training.';
exports.builder = function(yargs) {
  return yargs.option('project', {
    description:
      'The Google Cloud Platform project name to use for this invocation.',
    type: 'string'
    // required: true
  });
};
exports.handler = async function(argv) {
  try {
    tmp.setGracefulCleanup();

    const res = await downloadFile(LYRICS_FILE_URL);

    // const bucketName = 'tensorflow-lyrics-autocomplete';
    const bucketName = 'mr-lyrics-autocomplete-data';
    const storage = new Storage();
    const file = await uploadFile(storage, bucketName, res.name);

    // await storage.createBucket(bucketName);
    // console.log(`Bucket ${bucketName} created.`);
  } catch (err) {
    console.error(err);
  }
};
