from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import os
import tensorflow as tf


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------
# File paths
# --------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_recog_model_pwp.keras"
)

JSON_PATH = os.path.join(
    BASE_DIR,
    "plant_disease.json"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploadimages"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# --------------------------------------------------
# Load disease information
# --------------------------------------------------

with open(JSON_PATH, "r") as file:
    plant_disease = json.load(file)


# --------------------------------------------------
# Disease labels
# --------------------------------------------------

label = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


# --------------------------------------------------
# Serve uploaded images
# --------------------------------------------------

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def extract_features(image):
    image = tf.keras.utils.load_img(
        image,
        target_size=(160, 160)
    )

    feature = tf.keras.utils.img_to_array(image)

    feature = np.array([feature])

    return feature


# --------------------------------------------------
# Model prediction
# --------------------------------------------------

def model_predict(image):
    img = extract_features(image)

    prediction = model.predict(img, verbose=0)

    prediction_index = int(np.argmax(prediction))

    prediction_label = plant_disease[prediction_index]

    return prediction_label


# --------------------------------------------------
# Upload image and predict disease
# --------------------------------------------------

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():

    if request.method == "POST":

        # Check whether an image was submitted
        if 'img' not in request.files:
            return redirect(url_for('home'))

        image = request.files['img']

        # Check whether a file was actually selected
        if image.filename == '':
            return redirect(url_for('home'))

        # Generate a unique filename
        unique_name = f"temp_{uuid.uuid4().hex}_{image.filename}"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        # Save uploaded image
        image.save(image_path)

        print(f"Uploaded image: {image_path}")

        # Predict disease
        prediction = model_predict(image_path)

        # Path used by the HTML page to display the image
        image_url = url_for(
            'uploaded_images',
            filename=unique_name
        )

        return render_template(
            'home.html',
            result=True,
            imagepath=image_url,
            prediction=prediction
        )

    else:
        return redirect(url_for('home'))


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)