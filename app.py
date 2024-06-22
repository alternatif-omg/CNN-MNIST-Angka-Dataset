from flask import Flask, request, render_template, url_for
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the model
model = keras.models.load_model('mnist_model.h5')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    predicted_digit = None  # Initialize here
    image_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file to the server
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Process the image
            img = Image.open(filepath).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 pixels
            img = np.array(img)
            img = img.reshape(-1, 28, 28, 1)
            img = img / 255.0

            # Make prediction
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction[0])

            # URL for the uploaded image
            image_url = url_for('static', filename=file.filename)

    return render_template('index.html', prediction=predicted_digit, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
