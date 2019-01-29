const path = require('path');
const fs = require('fs');
const glob = require('glob');
// const { Storage } = require('@google-cloud/storage');
// Your Google Cloud Platform project ID
const projectId = 'YOUR_PROJECT_ID';
const BASE_LYRICS_TEXT_DIR = path.resolve('./lyrics/');

const files = glob.sync(path.join(BASE_LYRICS_TEXT_DIR, '/**/*.txt'));
let totalLength = 0;
const j = files.map(v => {
  let fileContents = fs.readFileSync(v).toString();
  const hasNewLine = fileContents.includes('\n');

  // if (hasNewLine) {
  //   fileContents = fileContents.replace(/\n/g, ' ');
  //   fs.writeFileSync(v, fileContents);
  // }

  totalLength += fileContents.length;

  return {
    filename: v,
    length: fileContents.length,
    hasNewLine: fileContents.includes('\n')
  };
});

console.log(j.filter(v => v.hasNewLine).length);
console.log(`total length ${totalLength}`);

// // Creates a client
// const storage = new Storage({
//   projectId: projectId
// });

// // The name for the new bucket
// const bucketName = 'my-new-bucket';

// // Creates the new bucket
// storage
//   .createBucket(bucketName)
//   .then(() => {
//     console.log(`Bucket ${bucketName} created.`);
//   })
//   .catch(err => {
//     console.error('ERROR:', err);
//   });
