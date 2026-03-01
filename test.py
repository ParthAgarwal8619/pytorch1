import torch
from torchvision import datasets, transforms
from models.cnn_model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("cnn_mnist_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST("./data", train=False, transform=transform)

image, label = test_dataset[0]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Actual:", label)
print("Predicted:", predicted.item())