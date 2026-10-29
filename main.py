from flask import Flask, request, jsonify, render_template
from model.mood_detection import MoodDetection
from model.music_recommender import MusicRecommender, MOOD_TARGETS


mood_detection = MoodDetection()
music_recommender = MusicRecommender()

# Static files (emoji images) are served from docs/ at /docs/<file>
app = Flask(__name__, static_folder='docs', static_url_path='/docs')

# Model emotion for each mood in the form's dropdown
MOOD_LABELS = {
    'Happy': 'happiness',
    'Sad': 'sadness',
    'Angry': 'anger',
    'Anxious': 'worry',
    'Excited': 'enthusiasm',
    'In love': 'love',
    'Relaxed': 'relief',
    'Bored': 'boredom',
    'Surprised': 'surprise',
    'Empty': 'empty',
    'Neutral': 'neutral',
}

# If the description gives the selected mood at least this score, the description supports the user's choice
AGREEMENT_THRESHOLD = 0.1

# The detected emotion always makes up at least this share of the blend, so the mix matches the headline emotion
MIN_LEAD_WEIGHT = 0.5


def build_blend(emotion, emotions):
    # Turns the description's top emotions into blend rates that add up to 1, e.g. {'anger': 0.6, 'sadness': 0.25, ...}.
    # If the detected emotion has less than MIN_LEAD_WEIGHT, it's raised to that and the others share the rest.
    total = sum(e['score'] for e in emotions)
    weights = {e['emotion']: e['score'] / total for e in emotions} if total else {}
    if weights.get(emotion, 0) < MIN_LEAD_WEIGHT:
        others = {mood: weight for mood, weight in weights.items() if mood != emotion}
        others_total = sum(others.values())
        weights = {mood: weight / others_total * (1 - MIN_LEAD_WEIGHT) for mood, weight in others.items()} if others_total else {}
        weights[emotion] = MIN_LEAD_WEIGHT if weights else 1.0
    return {mood: round(weight, 3) for mood, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)}


def parse_blend(blend):
    # Validates a blend from the request: {mood: non-negative number}. Returns (blend, error message)
    if blend is None:
        return None, None
    if not isinstance(blend, dict) or not blend:
        return None, 'blend must be an object like {"anger": 0.6, "sadness": 0.4}'
    for mood, weight in blend.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            return None, f'Rate for "{mood}" must be a number of 0 or more'
    return blend, None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    description = (data.get('description') or '').strip()
    if not description:
        return jsonify({'error': 'Please describe how you feel.'}), 400

    mood = data.get('mood')
    selected = MOOD_LABELS.get(mood)

    scores = mood_detection.mood_probabilities(description)
    detected = mood_detection.detect_mood(description, scores)

    # - description has no clear emotion (neutral is top): go with the selected mood
    # - description supports the selected mood: keep the user's choice
    # - description contradicts it: go with what the description says
    # Mixed feelings spread the scores across several moods, so this compares against the selected mood
    # instead of requiring one mood to dominate.
    if selected is None:
        emotion, source = detected, 'description'
    elif next(iter(scores)) == 'neutral':
        emotion, source = selected, 'selected'
    elif scores.get(selected, 0) >= AGREEMENT_THRESHOLD:
        emotion, source = selected, 'both'
    else:
        emotion, source = detected, 'description'

    # Insights: the strongest emotions in the description (neutral left out) and the sound the mix will aim for
    emotions = [{'emotion': name, 'score': round(float(score), 3)} for name, score in scores.items() if name != 'neutral'][:3]

    # Blend the mix from the emotions in the description, unless the description had no clear emotion
    blend = build_blend(emotion, emotions) if source != 'selected' else None

    return jsonify({
        'selected_mood': mood,
        'emotion': emotion,
        'source': source,
        'blend': blend,
        'insights': {
            'emotions': emotions,
            'sound': music_recommender.describe_sound(emotion, blend),
        },
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True) or {}
    emotion = data.get('emotion')
    if emotion not in MOOD_TARGETS:
        return jsonify({'error': f'Unknown emotion: {emotion}'}), 400

    # Optional blend rates, e.g. {"anger": 0.6, "sadness": 0.25, "happiness": 0.15}; without it the mix targets the emotion alone
    blend, error = parse_blend(data.get('blend'))
    if error:
        return jsonify({'error': error}), 400

    tracks = music_recommender.recommend_music(emotion, blend=blend)
    return jsonify({
        'emotion': emotion,
        'blend': blend,
        'target': music_recommender.get_target(emotion, blend),
        'tracks': tracks,
        'summary': music_recommender.summarize_mix(tracks),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
