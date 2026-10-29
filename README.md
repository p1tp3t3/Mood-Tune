# MoodTunes: Mood-Based Music Recommender

MoodTunes recommends music based on how you feel. Pick a mood, describe what's on your mind, and the app works out your emotion from your words. It then builds a 30-song mix that matches it, and you can play the songs right in the page.

## Features

- **Mood picker:** 11 moods to choose from, each shown as an emoji tile.
- **Emotion detection:** a pretrained language model reads your description and works out the emotion. It handles everyday sentences like "my dog died today", not just obvious words like "sad".
- **Mood vs. description:** if your description clearly contradicts the mood you picked, the description wins. If it supports your pick, or shows no clear emotion, your pick is used.
- **Insights after the analysis:**
  - the emotions found in your words
  - the sound the mix will aim for
  - why that kind of music suits the mood
  - a toggle to show or hide what you wrote
- **Recommendations:** 30 songs picked by nearest-neighbour search over the audio features of about 64,000 Spotify tracks.
- **Mix summary:** song count, total length, top genres, and how close the mix is to the mood's target sound.
- **Music player layout:** a track list with album covers and a player bar at the bottom that uses Spotify's embed player.

## How it works

### 1. Emotion detection ([model/mood_detection.py](model/mood_detection.py))

The description is classified by [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions), a RoBERTa model fine-tuned on GoEmotions (58,000 Reddit comments labelled by people with 28 emotions). Those 28 emotions are grouped into the 13 moods the app uses. For example, joy, optimism and pride all count as **happiness**, and fear, nervousness and embarrassment count as **worry**.

[main.py](main.py) then combines that result with the mood you picked:

| Description                                               | Result                                   |
| --------------------------------------------------------- | ---------------------------------------- |
| Shows no clear emotion (neutral comes out on top)         | Your picked mood                         |
| Supports your picked mood (that mood scores at least 10%) | Your picked mood                         |
| Contradicts your picked mood                              | The strongest emotion in the description |

### 2. Music recommendation ([model/music_recommender.py](model/music_recommender.py))

Each mood has a target sound defined by five audio features:

- **valence:** how positive the song sounds
- **energy**
- **danceability**
- **acousticness**
- **tempo**

For example, sadness targets low valence, low energy, high acousticness and about 80 BPM.

The app scales these features to the same range and uses scikit-learn's `NearestNeighbors` to find the 150 tracks closest to the target. It then picks 30 of those at random, so each new mix is different.

Before searching, the track data is cleaned:

- tracks listed more than once are removed (the same song appears once per genre it's tagged with, and re-releases repeat songs)
- tracks with a missing tempo are removed
- tracks with popularity under 20 are removed

Album covers come from Spotify's public oEmbed endpoint, and playback uses Spotify's embed player. Neither needs a Spotify API key.

## Running locally

Requires **Python 3.13** (other recent 3.x versions should also work) and about **2 GB of free RAM**.

```bash
pip install -r requirements.txt
python main.py
```

Then open http://localhost:5002.

The first start downloads the emotion model (about 500 MB) from Hugging Face. After that it loads from the local cache, and startup takes about 30 seconds.

> **Tip:** to save several GB, install the CPU-only build of PyTorch before installing the requirements:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Running with Docker

```bash
docker build -t moodtunes .
docker run -p 5002:5002 moodtunes
```

Then open http://localhost:5002.

The image includes the emotion model, so the container doesn't need to download it at startup. The app runs with gunicorn: one worker (each worker loads its own copy of the model) and four threads.

### Opening it from other devices on your network

The app listens on all network interfaces, so phones and other computers on the same Wi-Fi can open it at:

```
http://<this computer's LAN IP>:5002
```

On Windows, run `ipconfig` and use the IPv4 address under your Wi-Fi adapter. If other devices can't connect, Windows Firewall is probably blocking the port. Either set your Wi-Fi network to **Private**, or allow the port by running this in an Administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "MoodTunes 5002" -Direction Inbound -Protocol TCP -LocalPort 5002 -Action Allow -Profile Private,Public
```

## API

Both endpoints take and return JSON.

### `POST /predict`

Detects the emotion from the picked mood and a description.

```json
{ "mood": "Sad", "description": "my dog died today and I miss him so much" }
```

**Mood values:** `Happy`, `Sad`, `Angry`, `Anxious`, `Excited`, `In love`, `Relaxed`, `Bored`, `Surprised`, `Empty`, `Neutral`.

**Response fields:**

- `emotion`: the final mood.
- `source`: which input decided it.
  - `description`: the description overrode your pick.
  - `both`: the description supports your pick.
  - `selected`: the description had no clear emotion, so your pick was used.
- `insights`: the top emotions found in the description, and the target sound for the mood.

### `POST /recommend`

Builds a mix for a mood.

```json
{ "emotion": "sadness" }
```

**Emotion values:** `happiness`, `fun`, `enthusiasm`, `surprise`, `love`, `relief`, `neutral`, `boredom`, `worry`, `anger`, `hate`, `sadness`, `empty`.

**Response fields:**

- `tracks`: 30 songs, each with title, artists, album, genre, duration, audio features, album cover and Spotify links.
- `summary`: average audio features, top genres and total length of the mix.

## Limitations

- **Mixed feelings:** the model picks one main emotion. For text with mixed feelings, it may not choose the one you'd expect.
- **Love:** the model often reads affectionate text as joy or admiration rather than love.
- **Same sound, different styles:** recommendations match how a song sounds, not its genre or language, so one mix can combine very different styles.
- **Internet needed:** album covers and playback require a connection. The Spotify player plays 30-second previews unless you're logged in to Spotify in the browser.

## Credits

- **Emotion model:** [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions), trained on Google's [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset.
- **Track data:** a Spotify tracks dataset from Kaggle (114,000 tracks across 114 genres, with Spotify audio features).
- **Emoji images:** [Noto Emoji](https://github.com/googlefonts/noto-emoji) by Google (Apache License 2.0).
- **Playback and album covers:** Spotify's embed player and oEmbed endpoint.
