import re
import io
import base64
import torch
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from model import MNISTModel # your trained model class
#from model import MNIST_CNN
from torchvision.transforms import ToPILImage

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

def preprocess_canvas_image(img):    

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Invert colors: MNIST digits are white (255) on black (0)
    img = ImageOps.invert(img)

    # Resize to 28x28 if needed
    img = img.resize((28, 28))

    # Convert to numpy array and scale pixels 0-1
    img_arr = np.array(img) / 255.0

    # Flatten to 784 vector
    img_arr = img_arr.reshape(1, 784).astype(np.float32)

    # Convert to tensor
    tensor = torch.from_numpy(img_arr)

    return tensor

@app.route('/predict', methods=['POST'])
def predict():
    from PIL import ImageOps
    import numpy as np
    from torchvision.transforms import ToTensor, Normalize, Compose

    data = request.get_json()
    img_data = data['image']
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    image_bytes = base64.b64decode(img_str)

    # Step 1: Load and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Step 2: Invert the image (black bg, white digit like MNIST)
    image = ImageOps.invert(image)

    # Step 3: Crop the content area to remove whitespace
    np_img = np.array(image)
    coords = np.argwhere(np_img < 255)  # non-white pixels
    if coords.size == 0:
        return jsonify({'prediction': 'blank'})

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    image = image.crop((x0, y0, x1, y1))  # (left, upper, right, lower)

    # Step 4: Resize to 20x20 (like MNIST) and paste into 28x28 center
    image = image.resize((20, 20), Image.Resampling.LANCZOS)
    new_image = Image.new('L', (28, 28), 0)  # black background
    new_image.paste(image, ((28 - 20) // 2, (28 - 20) // 2))

    # Optional: Save debug image to verify it's centered
    new_image.save("debug_final_input.png")

    # Step 5: Convert to tensor and normalize
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = transform(new_image).unsqueeze(0)  # shape [1,1,28,28]

    input_tensor = input_tensor.view(-1, 28 * 28)     # shape [1, 784]

    # Predict
    with torch.no_grad():
     output = model(input_tensor)
     prediction = output.argmax(dim=1).item()
     

    return jsonify({'prediction': prediction})



if __name__ == '__main__':
    app.run(debug=True)