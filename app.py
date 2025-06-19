from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model("ssrnet_3_3_3_64_1.0_1.0.h5")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((64, 64))  # Kích thước cho SSR-Net
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_bytes = img_file.read()
    img = preprocess_image(img_bytes)

    age = model.predict(img)[0][0]
    label = "child" if age < 12 else "adult"
    return jsonify({"age_estimate": float(age), "label": label})
