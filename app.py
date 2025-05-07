import os
import torch
import timm
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
def load_model(model_path, num_classes):
    model = timm.create_model("efficientnet_b3a", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model_path = "plant_best_model (1).pth"
model = load_model(model_path, num_classes=len(class_names))

# Transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Predict function
def predict_image(file):
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return class_names[pred.item()]

@app.route("/", methods=["GET"])
def home():
    return "Plant Classification API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "imagefile" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["imagefile"]
    try:
        prediction = predict_image(file)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=3000, debug=True)
