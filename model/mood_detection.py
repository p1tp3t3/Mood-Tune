from transformers import pipeline

# Minimum confidence needed before 'neutral' is returned over the next most likely mood
NEUTRAL_THRESHOLD = 0.5

# Pretrained RoBERTa model fine-tuned on GoEmotions (58k human-labelled Reddit comments, 28 emotions).
# It understands descriptions like "my dog died today", not just explicit emotion words.
PRETRAINED_MODEL = 'SamLowe/roberta-base-go_emotions'

# GoEmotions labels mapped onto the moods MusicRecommender knows
GO_EMOTIONS_LABELS = {
    'joy': 'happiness', 'optimism': 'happiness', 'pride': 'happiness', 'gratitude': 'happiness', 'admiration': 'happiness',
    'amusement': 'fun',
    'excitement': 'enthusiasm',
    'love': 'love', 'caring': 'love', 'desire': 'love',
    'relief': 'relief',
    'surprise': 'surprise', 'realization': 'surprise', 'curiosity': 'surprise',
    'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
    'disgust': 'hate',
    'sadness': 'sadness', 'grief': 'sadness', 'disappointment': 'sadness', 'remorse': 'sadness',
    'fear': 'worry', 'nervousness': 'worry', 'embarrassment': 'worry', 'confusion': 'worry',
    'approval': 'neutral', 'neutral': 'neutral',
}


class MoodDetection:
    def __init__(self):
        # Downloaded (~500 MB) from Hugging Face on first run, then loaded from the local cache
        self.model = pipeline('text-classification', model=PRETRAINED_MODEL, top_k=None, truncation=True)

    def detect_mood(self, text, scores=None):
        # Returns the most likely mood for the given text (pass scores from mood_probabilities to skip re-running the model)
        if not text or not text.strip():
            return 'neutral'

        scores = scores or self.mood_probabilities(text)
        top_mood, top_prob = next(iter(scores.items()))

        # Only trust a 'neutral' prediction when the model is confident about it
        if top_mood == 'neutral' and top_prob < NEUTRAL_THRESHOLD:
            return next(mood for mood in scores if mood != 'neutral')
        return top_mood

    def mood_probabilities(self, text):
        # Returns {mood: probability} sorted from most to least likely.
        # GoEmotions scores are independent per label, so sum them per mood and normalise to add up to 1.
        scores = {}
        for result in self.model(text)[0]:
            mood = GO_EMOTIONS_LABELS[result['label']]
            scores[mood] = scores.get(mood, 0) + result['score']
        total = sum(scores.values())
        scores = {mood: score / total for mood, score in scores.items()}
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


if __name__ == '__main__':
    detector = MoodDetection()
    for sample in ["I just got the job, I can't believe it!", "I miss her so much", "ugh this traffic is making me furious"]:
        print(sample, '->', detector.detect_mood(sample))
