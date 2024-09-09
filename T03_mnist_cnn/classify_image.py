import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from mnist_cnn import Net

model = Net()

# Check if the model weights are available
model_path = "mnist_cnn.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
else:
    print("Error: Model weights not found! Please train the model and save the weights first.")
    print("To train the model and save the weights, run the following command:")
    print("    python3 mnist_cnn.py --save-model")
    exit(1)
model.eval()

# Preprocess the image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def classify_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Recognizer, enter the file name of a picture')
    parser.add_argument('image', type=str, help='the file name of the picture')
    args = parser.parse_args()
    image_path = args.image
    prediction = classify_image(image_path)
    print(f'The predicted class is: {prediction}')

if __name__ == '__main__':
    main()