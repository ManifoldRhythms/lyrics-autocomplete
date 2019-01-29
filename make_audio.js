const textToSpeech = require('@google-cloud/text-to-speech');
const fs = require('fs');
const path = require('path');
const util = require('util');

const client = new textToSpeech.TextToSpeechClient();

const ssml = fs.readFileSync('./predictions/ssml.xml');
const outputFile = path.resolve('./output/output.mp3');

async function generate() {
  const request = {
    input: { ssml: ssml },
    voice: {
      languageCode: 'en-AU',
      ssmlGender: 'MALE',
      voiceName: 'en-AU-Wavenet-B'
    },
    audioConfig: { audioEncoding: 'MP3' }
  };

  const [response] = await client.synthesizeSpeech(request);
  const writeFile = util.promisify(fs.writeFile);
  await writeFile(outputFile, response.audioContent, 'binary');
  console.log(`Audio content written to file: ${outputFile}`);
}

generate().catch(console.error);
