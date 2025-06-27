from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('vgg16.h5')

@app.route('/')

def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("aboutus.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/predict-page')
def predict_page():
    return render_template("predict.html")

@app.route('/predict', methods=["POST"])
def predict():
    f = request.files['file']
    img_path = os.path.join("static/uploads", f.filename)
    f.save(img_path)

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)
    labels = ["Biodegradable Images (0)", "Recyclable Images (1)", "Trash Images (2)"]
    prediction = labels[np.argmax(pred)]

    return render_template("portfolio-details.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True, port=5222)
