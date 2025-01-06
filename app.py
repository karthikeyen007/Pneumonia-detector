from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = load_model('pneumonia_model.h5')

@app.route('/')
def welcome():
    return render_template('welcome.html')  # Serve the welcome page

@app.route('/main')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file:
                # Secure the file name and save it in static folder
                filename = secure_filename(file.filename)
                filepath = os.path.join('static', filename)
                file.save(filepath)

                # Preprocess the image for prediction
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Make the prediction
                prediction = model.predict(img_array)
                class_label = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

                return jsonify({'classification': class_label})
            else:
                return jsonify({'error': 'No file uploaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
