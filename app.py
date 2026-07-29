from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# -----------------------------
# PlantVillage Classes
# Replace this list with your COMPLETE 38-class list
# -----------------------------
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template("index.html")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html")

    image = Image.open(file)
    image = preprocess(image)

    prediction = model.predict(image, verbose=0)

    class_index = np.argmax(prediction)

    confidence = round(float(np.max(prediction)) * 100, 2)

    result = classes[class_index]

    if "___" in result:
        crop, disease = result.split("___")
    else:
        crop = result
        disease = "Healthy"

    crop = crop.replace("_", " ")
    crop = crop.replace(",", "")

    disease = disease.replace("_", " ")

    disease_data = {

        "Apple scab": (
            "Apply Captan or Mancozeb fungicide.",
            "Prune infected leaves and maintain orchard sanitation."
        ),

        "Black rot": (
            "Spray Copper fungicide.",
            "Remove infected fruits and leaves."
        ),

        "Cedar apple rust": (
            "Apply fungicide during spring.",
            "Avoid planting cedar trees near apple orchards."
        ),

        "Powdery mildew": (
            "Spray Sulfur fungicide.",
            "Avoid excess humidity."
        ),

        "Common rust": (
            "Apply Propiconazole fungicide.",
            "Grow resistant varieties and monitor regularly."
        ),

        "Northern Leaf Blight": (
            "Use Mancozeb fungicide.",
            "Follow crop rotation."
        ),

        "Early blight": (
            "Spray Chlorothalonil.",
            "Remove infected leaves."
        ),

        "Late blight": (
            "Spray Metalaxyl.",
            "Avoid excessive moisture."
        ),

        "Leaf Mold": (
            "Improve air circulation.",
            "Reduce humidity."
        ),

        "Septoria leaf spot": (
            "Apply Copper fungicide.",
            "Avoid overhead irrigation."
        ),

        "Spider mites": (
            "Use Miticide spray.",
            "Maintain field hygiene."
        ),

        "Target Spot": (
            "Apply fungicide.",
            "Avoid water stagnation."
        ),

        "Tomato mosaic virus": (
            "Remove infected plants.",
            "Disinfect farming tools."
        ),

        "Tomato Yellow Leaf Curl Virus": (
            "Control whiteflies.",
            "Use virus-resistant varieties."
        ),

        "Bacterial spot": (
            "Apply Copper bactericide.",
            "Use disease-free seeds."
        ),

        "healthy": (
            "No treatment required.",
            "Continue good agricultural practices."
        )
    }

    treatment = "Consult an Agricultural Officer."

    prevention = "Maintain proper crop management."

    for key in disease_data:

        if key.lower() in disease.lower():

            treatment = disease_data[key][0]

            prevention = disease_data[key][1]

            break

    return render_template(
        "index.html",
        prediction=result,
        crop=crop,
        disease=disease,
        confidence=confidence,
        treatment=treatment,
        prevention=prevention
    )


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)