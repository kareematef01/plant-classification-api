import torch
import timm
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image

# Flask setup
app = Flask(__name__)

# Load model function
def load_model(model_path, num_classes):
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load your trained model
model_path = "plant_best_model(1).pth"  
num_classes = 30
model = load_model(model_path, num_classes)

# Class names: النباتات فقط
class_names = [
    "aloevera", "banana", "bilimbi", "cantaloupe", "cassava", "coconut",
    "corn", "cucumber", "curcuma", "eggplant", "galangal", "ginger",
    "guava", "kale", "longbeans", "mango", "melon", "orange", "paddy",
    "papaya", "peperchili", "pineapple", "pomelo", "shallot", "soybeans",
    "spinach", "sweetpotatoes", "tobacco", "waterapple", "watermelon"
]

# Transformations
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(file):
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return class_names[predicted.item()]

# API endpoint
@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['imagefile']
    try:
        prediction = predict_image(file)
        return jsonify({"result": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app locally
if __name__ == '__main__':
    app.run(port=8000, debug=True)
