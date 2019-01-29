/**
 * Track info.
 * @typedef {Object} Track
 * @property {number|string} track_id
 * @property {string} track_name
 * @property {number} has_lyrics
 * @property {number|string} album_id
 * @property {string} album_name
 * @property {number|string} artist_id
 * @property {string} artist_name
 */
/**
 * Lyrics of a track.
 * @typedef {Object} Lyrics
 * @property {number|string} lyrics_id
 * @property {string} lyrics_body
 * @property {string} updated_time
 * @property {number} explicit
 */

const API_KEY = process.env.API_KEY;
const path = require('path');
const fs = require('fs');
const glob = require('glob');
const rp = require('request-promise').defaults({
  json: true,
  baseUrl: 'https://api.musixmatch.com/ws/1.1/',
  qs: {
    apikey: API_KEY
  }
});
const colors = require('colors');

/**
 *
 * @param {Object} [options]
 * @param {number} [options.page=1]
 * @param {number} [options.page_size=10]
 * @param {string} [options.country=us]
 * @param {boolean} [options.has_lyrics=true]
 *
 * @returns {Promise<Track[]>}
 */
async function listChart(
  options = {
    format: 'json',
    has_lyrics: true,
    page: 1,
    page_size: 10,
    country: 'us'
  }
) {
  const _options = {
    format: 'json',
    page: '1',
    page_size: '10',
    country: 'us',
    f_has_lyrics: options.has_lyrics === false ? '0' : '1',
    ...options
  };
  try {
    const response = await rp.get('/chart.tracks.get', {
      qs: _options
    });
    const tracks = response.message.body.track_list;
    // fs.writeFileSync('./log/lyrics.json', JSON.stringify(tracks, null, 2));
    return tracks;
  } catch (err) {
    console.error(err);
  }
}

/**
 *
 * @param {number|string} trackId
 * @returns {Lyrics}
 */
async function getLyrics(trackId) {
  try {
    const response = await rp.get('/track.lyrics.get', {
      qs: {
        format: 'json',
        track_id: trackId
      }
    });
    const lyrics = response.message.body.lyrics;
    // fs.writeFileSync(
    //   `./log/lyrics_${trackId}.json`,
    //   JSON.stringify(lyrics, null, 2)
    // );
    return lyrics;
  } catch (err) {
    console.error(err);
  }
  return null;
}

/**
 *
 * @param {Object[]} tracks
 * @param {Track} tracks[].track
 */
async function batchGetLyrics(tracks) {
  const response = [];

  for (const track of tracks) {
    try {
      const lyrics = await getLyrics(track.track.track_id);
      if (lyrics) {
        response.push(lyrics);
      }
    } catch (err) {
      console.error(err);
    }
  }

  return response;
}

function listAllSongs() {
  const BASE_CHART_LIST_DIR = path.resolve('./log/');
  const files = glob.sync(
    path.join(BASE_CHART_LIST_DIR, 'chart_list_page*.json')
  );

  for (const filename of files) {
    /**
     * @type {{track:Track}[]} data
     */
    const data = JSON.parse(fs.readFileSync(filename)).sort((a, b) => {
      return a.track.artist_name.localeCompare(b.track.artist_name);
    });

    // console.log('loading batch', filename, `- ${data.length} records`);

    for (const { track } of data) {
      console.log(
        // `${track.album_name} - ${track.artist_name} - ${track.track_name}`
        `${track.artist_name.bold} - ${track.track_name}`
      );
    }
  }
}

module.exports = {
  listChart
};

// listChart({
//   page: 3,
//   page_size: 100
// }).then(res => {
//   fs.writeFileSync(
//     // './log/chart_list_page_0001.json',
//     // './log/chart_list_page_0002.json',
//     './log/chart_list_page_0003.json',
//     JSON.stringify(res, null, 2)
//   );
// });

// // const chartList = require('./log/chart_list_page_0001.json');
// // const chartList = require('./log/chart_list_page_0002.json');
// const chartList = require('./log/chart_list_page_0003.json');

// batchGetLyrics(chartList).then(res => {
//   fs.writeFileSync(
//     // `./log/lyrics_batch_page_0001.json`,
//     // `./log/lyrics_batch_page_0002.json`,
//     `./log/lyrics_batch_page_0003.json`,
//     JSON.stringify(res, null, 2)
//   );
// });

// listAllSongs();
