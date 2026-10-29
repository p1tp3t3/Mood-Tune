import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKS_PATH = os.path.join(BASE_DIR, '..', 'spotify-tracks.csv')
OEMBED_URL = 'https://open.spotify.com/oembed'

FEATURES = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

# Target audio profile for each mood detected by MoodDetection.
# valence (positivity), energy, danceability, acousticness are 0-1; tempo is in BPM.
MOOD_TARGETS = {
    'happiness':  {'valence': 0.85, 'energy': 0.75, 'danceability': 0.70, 'acousticness': 0.20, 'tempo': 120},
    'fun':        {'valence': 0.80, 'energy': 0.80, 'danceability': 0.80, 'acousticness': 0.10, 'tempo': 124},
    'enthusiasm': {'valence': 0.75, 'energy': 0.90, 'danceability': 0.65, 'acousticness': 0.05, 'tempo': 135},
    'surprise':   {'valence': 0.65, 'energy': 0.80, 'danceability': 0.60, 'acousticness': 0.15, 'tempo': 130},
    'love':       {'valence': 0.65, 'energy': 0.45, 'danceability': 0.55, 'acousticness': 0.50, 'tempo': 100},
    'relief':     {'valence': 0.60, 'energy': 0.35, 'danceability': 0.50, 'acousticness': 0.65, 'tempo': 95},
    'neutral':    {'valence': 0.50, 'energy': 0.55, 'danceability': 0.55, 'acousticness': 0.35, 'tempo': 115},
    'boredom':    {'valence': 0.35, 'energy': 0.30, 'danceability': 0.45, 'acousticness': 0.50, 'tempo': 95},
    'worry':      {'valence': 0.25, 'energy': 0.45, 'danceability': 0.40, 'acousticness': 0.55, 'tempo': 100},
    'anger':      {'valence': 0.25, 'energy': 0.92, 'danceability': 0.45, 'acousticness': 0.05, 'tempo': 140},
    'hate':       {'valence': 0.20, 'energy': 0.95, 'danceability': 0.40, 'acousticness': 0.03, 'tempo': 150},
    'sadness':    {'valence': 0.15, 'energy': 0.25, 'danceability': 0.35, 'acousticness': 0.75, 'tempo': 80},
    'empty':      {'valence': 0.10, 'energy': 0.15, 'danceability': 0.25, 'acousticness': 0.80, 'tempo': 75},
}

# Skip very obscure tracks (16k tracks in the dataset have popularity 0)
MIN_POPULARITY = 20

# Recommendations are sampled from this many times n nearest tracks, so the same mood doesn't always return the same list
CANDIDATE_MULTIPLIER = 5


def blend_targets(weights):
    # Weighted average of mood targets, e.g. {'anger': 0.6, 'sadness': 0.4} -> a target between the two.
    # Unknown moods and non-positive weights are ignored; returns None if nothing usable is left.
    usable = {mood: float(weight) for mood, weight in weights.items() if mood in MOOD_TARGETS and weight > 0}
    total = sum(usable.values())
    if not total:
        return None
    return {
        feature: round(sum(MOOD_TARGETS[mood][feature] * weight for mood, weight in usable.items()) / total, 2)
        for feature in FEATURES
    }


def describe_features(features):
    # Turns audio features into a short description like "Upbeat, high-energy songs around 120 BPM, mostly electric"
    valence, energy, acousticness = features['valence'], features['energy'], features['acousticness']
    mood_word = 'Upbeat' if valence >= 0.65 else 'Melancholic' if valence <= 0.35 else 'Balanced'
    energy_word = 'high-energy' if energy >= 0.7 else 'calm' if energy <= 0.35 else 'mid-energy'
    sound_word = 'mostly acoustic' if acousticness >= 0.6 else 'mostly electric' if acousticness <= 0.2 else 'a mix of acoustic and electric'
    return f"{mood_word}, {energy_word} songs around {round(features['tempo'])} BPM, {sound_word}"


class MusicRecommender:
    def __init__(self):
        self.tracks = self.load_tracks(TRACKS_PATH)
        self.scaler = MinMaxScaler()
        features = self.scaler.fit_transform(self.tracks[FEATURES])
        self.knn = NearestNeighbors(metric='euclidean').fit(features)

    def load_tracks(self, file_path):
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        df = df.dropna(subset=['track_name', 'artists'])
        df = df[(df['tempo'] > 0) & (df['popularity'] >= MIN_POPULARITY)]

        # The same track appears once per genre it's tagged with, and re-releases share a name and artist
        df = df.sort_values('popularity', ascending=False)
        df = df.drop_duplicates(subset='track_id')
        df = df.drop_duplicates(subset=['track_name', 'artists'])
        return df.reset_index(drop=True)

    def get_target(self, mood, blend=None):
        # Target audio profile: a blend of several moods if given, otherwise the single mood's target
        return (blend and blend_targets(blend)) or MOOD_TARGETS.get(mood, MOOD_TARGETS['neutral'])

    def recommend_music(self, mood, n=30, blend=None):
        # Returns n tracks whose audio features are closest to the target profile for the mood (or blend of moods)
        target = self.get_target(mood, blend)
        target = self.scaler.transform(pd.DataFrame([target], columns=FEATURES))

        _, indices = self.knn.kneighbors(target, n_neighbors=n * CANDIDATE_MULTIPLIER)
        picks = np.random.choice(indices[0], size=n, replace=False)
        results = self.tracks.iloc[picks].sort_values('popularity', ascending=False)

        tracks = [
            {
                'track_name': row.track_name,
                'artists': row.artists.replace(';', ', '),
                'album_name': row.album_name,
                'genre': row.track_genre,
                'duration_ms': int(row.duration_ms),
                'features': {feature: round(float(getattr(row, feature)), 2) for feature in FEATURES},
                # Spotify's embed player needs no API key; it plays a preview with cover art (full track if logged in to Spotify)
                'embed_url': f'https://open.spotify.com/embed/track/{row.track_id}',
                'track_url': f'https://open.spotify.com/track/{row.track_id}',
                'spotify_uri': f'spotify:track:{row.track_id}',
            }
            for row in results.itertuples()
        ]

        # Fetch covers in parallel so 10 tracks take about as long as one request
        with ThreadPoolExecutor(max_workers=len(tracks) or 1) as pool:
            for track, image in zip(tracks, pool.map(self.get_album_image, tracks)):
                track['album_image'] = image
        return tracks

    def describe_sound(self, mood, blend=None):
        # Target audio profile plus a short plain-English summary, e.g. "Upbeat, high-energy songs around 120 BPM"
        target = self.get_target(mood, blend)
        return {'features': target, 'summary': describe_features(target)}

    def summarize_mix(self, tracks):
        # Average audio features, top genres and total length of a recommended mix
        if not tracks:
            return None
        averages = {
            feature: round(sum(track['features'][feature] for track in tracks) / len(tracks), 2)
            for feature in FEATURES
        }
        genres = pd.Series([track['genre'] for track in tracks]).value_counts()
        return {
            'features': averages,
            'summary': describe_features(averages),
            'top_genres': genres.index[:3].str.replace('-', ' ').tolist(),
            'total_ms': sum(track['duration_ms'] for track in tracks),
        }

    def get_album_image(self, track):
        # Spotify's oEmbed endpoint is public (no API key or Premium needed); covers are optional, so failures return None
        try:
            response = requests.get(OEMBED_URL, params={'url': track['track_url']}, timeout=4)
            response.raise_for_status()
            return response.json().get('thumbnail_url')
        except (requests.RequestException, ValueError):
            return None