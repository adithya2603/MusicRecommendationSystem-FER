from flask import Flask, render_template, request
import cv2
from detect_emotion import detect_emotion
from recommend_music import recommend_music

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    emotion = detect_emotion()
    song = recommend_music(emotion)
    return render_template("index.html", emotion=emotion, song=song)

if __name__ == "__main__":
    app.run(debug=True)
