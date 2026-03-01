import tkinter as tk
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from models.cnn_model import CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_mnist_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw Digit - CNN Predictor")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label = tk.Label(root, text="Draw a digit", font=("Arial", 18))
        self.label.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label.config(text="Draw a digit")

    def predict(self):
        img = transform(self.image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        self.label.config(text=f"Prediction: {predicted.item()}")

root = tk.Tk()
app = DigitApp(root)
root.mainloop()