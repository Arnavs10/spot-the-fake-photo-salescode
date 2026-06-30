"""
app.py -- tiny live camera demo (optional, the "impressive" extra).

  pip install flask
  python app.py
  open http://127.0.0.1:5000

Serves a page that grabs webcam frames and shows a live real/screen score.
Point your webcam at a normal object, then at a screen showing a photo, and
watch the score swing. Reuses the SAME features.py + model.pkl as predict.py.
"""
import io
import base64
from flask import Flask, request, jsonify, send_file
from PIL import Image

from features import extract_features
from predict import _model

app = Flask(__name__)
_model()  # warm up at startup


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)["image"]
    raw = base64.b64decode(data.split(",", 1)[1])  # strip data:image/...;base64,
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    score = float(_model().predict_proba(extract_features(img).reshape(1, -1))[0, 1])
    return jsonify({"score": score})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
