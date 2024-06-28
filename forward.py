import numpy as np
from PIL import Image
import torch

# Number of top predictions to display
top_k = 10

# Load ImageNet labels
with open("classes.txt") as f:
    imagenet_labels = {i: line.strip() for i, line in enumerate(f)}

# Load the pre-trained model
model = torch.load("model.pth")
model.eval()

# Load and preprocess the image
image = Image.open("converted_image.png")
image_array = np.array(image)
image_array = (image_array / 128) - 1  # Normalize to range [-1, 1]

# Convert image to tensor and prepare for model input
input_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()

# Perform inference
logits = model(input_tensor)
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get top-k predictions
top_probabilities, top_indices = probabilities[0].topk(top_k)

# Print top-k predictions with probabilities
for rank, (index, probability) in enumerate(zip(top_indices, top_probabilities)):
    class_index = index.item()
    class_probability = probability.item()
    class_label = imagenet_labels[class_index]
    print(f"{rank}: {class_label:<45} --- {class_probability:.4f}")
