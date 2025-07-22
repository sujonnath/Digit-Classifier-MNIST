from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import re
from model import MNISTModel  # your trained model class

app = Flask(__name__)

# Load the trained model
model = MNISTModel()
model.load_state_dict(torch.load('model/mnist_model.pt', map_location='cpu'))
model.eval()

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(),          # Ensure it's grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']

    # Decode base64 string
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    image_bytes = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Invert and normalize image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)