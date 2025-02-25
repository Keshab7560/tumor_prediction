from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import cv2
import joblib
import requests
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Hugging Face model file URL (Updated with 'resolve')
MODEL_URL = "https://huggingface.co/Keshab7560/tumor-detection-model/resolve/main/model.pkl"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Ensure model folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model if it doesn’t exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as model_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                model_file.write(chunk)
    print("Model downloaded successfully.")

# Load the model
model = joblib.load(MODEL_PATH)

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        name = request.form["name"]
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128)).flatten().reshape(1, -1)  # Flatten for ML model

            # Predict
            prediction = model.predict(img)[0]
            result = "Tumor Detected" if prediction == 1 else "Normal"

            # Save result in Excel
            df = pd.DataFrame([[name, filename, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
                              columns=["Patient Name", "Image", "Result", "Date"])
            excel_path = "results.xlsx"

            if os.path.exists(excel_path):
                existing_df = pd.read_excel(excel_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_excel(excel_path, index=False)

            return redirect(url_for("result", name=name, result=result, image=filename))

    return render_template("upload.html")

@app.route("/result")
def result():
    name = request.args.get("name")
    result = request.args.get("result")
    image = request.args.get("image")
    return render_template("result.html", name=name, result=result, image=image)

if __name__ == "__main__":
    app.run(debug=True)
