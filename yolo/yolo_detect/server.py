from flask import Flask, request, jsonify
from _wsgi import get_model

app = Flask(__name__)
model = get_model()

@app.route("/predict", methods=["POST"])
def predict():
    tasks = request.json["tasks"]
    preds = model.predict(tasks)
    return jsonify(preds)

@app.route("/train", methods=["POST"])
def train():
    tasks = request.json["tasks"]
    annotations = request.json["annotations"]
    result = model.fit(tasks, annotations)
    return jsonify(result)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "detail": "YOLO11SegModel Backend Running"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"})

@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
