import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('E:/project/skin_cancer_detection_model.h5')

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected.")
        
        if file:
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            img_array = preprocess_image(image_path)
            prediction_score = model.predict(img_array)[0][0]
            prediction = "Benign (Non-cancerous)" if prediction_score < 0.5 else "Malignant (Cancerous)"

    return render_template('index.html', prediction=prediction, image_path=image_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)