🧠 MNIST Digit Classifier

A simple web app to draw handwritten digits (0-9) and recognize them using a trained PyTorch model.

 ![Screenshot](https://github.com/sujonnath/Digit-Classifier-MNIST/blob/main/2025-07-22%2017_37_56-Window.png)
 ![Screenshot](https://github.com/sujonnath/Digit-Classifier-MNIST/blob/main/2025-07-22%2017_54_33-Window.png).

---

📌 Features

- Draw digits on an interactive canvas
- Predict the digit using a neural network trained on MNIST dataset
- Supports mouse and touch input
- Clear canvas and redraw
- Shows prediction results instantly

---

📂 Project Structure
```
digitclassifier/
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   ├── index.html
│   └── train.html
├── model/
│   └── mnist_model.pt
├── debug_final_input.png
├── app.py
├── model_builder.py
└── README.md
```
⚙️ Installation
```

1. Clone the repository:

git clone https://github.com/sujonnath/Digit-Classifier-MNIST.git
cd mnist-digit-classifier

2. Create and activate a virtual environment (optional but recommended)
python -m venv venv 
venv\Scripts\activate # Windows
source venv/bin/activate # macOS/Linux

3. Install dependencies
pip install -r requirements.txt
pip install torch torchvision flask pillow numpy

4. Train the model (optional)
If you want to retrain the model on the MNIST dataset:

python model.py
This will train the model and save the weights to model/mnist_model.pt.

5. Run the Flask app:
python app.py

```

🧠 Model Summary

    Input: 28x28 grayscale images (flattened or CNN input)
    Hidden Layers: 2–3 dense layers with ReLU activation
    Batch Normalization and Dropout for regularization
    Output: 10-class Softmax (digits 0–9)

📦 Requirements


    Python 3.8+ (>= 3.11.9)
    TensorFlow 2.x/torch
    Flask
    Matplotlib
    NumPy




