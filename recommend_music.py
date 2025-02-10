import random

music_dict = {
    "Happy": ["happy_song1.mp3", "happy_song2.mp3"],
    "Sad": ["sad_song1.mp3", "sad_song2.mp3"],
    "Angry": ["angry_song1.mp3", "angry_song2.mp3"],
    "Neutral": ["neutral_song1.mp3", "neutral_song2.mp3"],
    "Surprise": ["surprise_song1.mp3", "surprise_song2.mp3"],
    "Fear": ["fear_song1.mp3", "fear_song2.mp3"],
    "Disgust": ["disgust_song1.mp3", "disgust_song2.mp3"]
}

def recommend_music(emotion):
    return random.choice(music_dict.get(emotion, []))
