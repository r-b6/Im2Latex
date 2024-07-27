from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2 as cv
import numpy as np
import torch
import pandas as pd
from model import Model
from decode import LatexProducer

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 100
MODEL_PATH = './utils/best_ckpt.pt'
VOCAB_PATH = './utils/dict_id2word.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = pd.read_pickle(VOCAB_PATH)
model = Model(device, len(vocab), 32)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
producer = LatexProducer(model, vocab)
model.eval()


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/create', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'error: no file'})
        file = request.files['image']
        if file is None or file.filename == "":
            return jsonify({'error': 'error: no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'error: format not supported'})

        try:
            with torch.no_grad():
                tensor = transform_image(file)
                prediction = producer(tensor) #returns a tuple with
                #second element as log probs, which we don't need for inference
                prediction = prediction[0]
                prediction = producer.get_formulas(prediction)
                data = {'prediction': prediction}
            return jsonify(data)
        except Exception as error:
            return jsonify({'error': 'error during prediction'})


def transform_image(image_file):
    """
    :param image_file: DataStorage object
    :return: torch tensor of dimension [1, H, W]
    """
    #transforming input into np array
    image_bytes = image_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    #preprocessing image
    image = cv.imdecode(np_arr, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT),
                      interpolation=cv.INTER_AREA)
    image = np.expand_dims(image, 0)
    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0)
    return tensor


if __name__ == '__main__':
    app.run(debug=True)
