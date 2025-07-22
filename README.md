🧠 MNIST Digit Classifier

A simple web app to draw handwritten digits (0-9) and recognize them using a trained PyTorch model.

---

📌 Features

- Draw digits on an interactive canvas
- Predict the digit using a neural network trained on MNIST dataset
- Supports mouse and touch input
- Clear canvas and redraw
- Shows prediction results instantly

---

📂 Project Structure
Digit-Classifier-MNIST/
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
├── mnist_model.py
├── README.md

⚙️ Installation

1. Clone the repository:

git clone https://github.com/aronno1920/mnist-digit-classifier.git
cd mnist-digit-classifier
2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
pip install torch torchvision flask pillow numpy

4. Train the model (optional)
If you want to retrain the model on the MNIST dataset:

python model.py
This will train the model and save the weights to model/mnist_model.pt.





