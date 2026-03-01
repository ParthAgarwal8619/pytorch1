from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
from models.cnn_model import CNN

app = Flask(__name__)

device = torch.device("cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("cnn_mnist_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    image = Image.open(file).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence_tensor, predicted = torch.max(probabilities, 1)

    return jsonify({
        "prediction": int(predicted.item()),
        "confidence": round(confidence_tensor.item() * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)