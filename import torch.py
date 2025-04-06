import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from flask import Flask, request, jsonify, render_template

# === Model Paths ===
MODEL_PATHS = {
    'dr': 'models/dr_model.pth',
    'brain_tumor': 'models/brain_tumor.pth',
}

# === Model Loading Functions ===

def load_dr_model(model_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_brain_tumor_model(model_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=4, bias=True)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_model(model_type):
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Model type '{model_type}' not supported")
    
    model_path = MODEL_PATHS[model_type]
    
    if model_type == 'dr':
        return load_dr_model(model_path)
    elif model_type == 'brain_tumor':
        return load_brain_tumor_model(model_path)

# === Prediction Functions ===

def predict_dr_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
    return classes[predicted.item()]

def predict_brain_tumor_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return classes[predicted.item()]

def predict(model_type, input_data):
    model = load_model(model_type)
    
    if model_type == 'dr':
        return predict_dr_image(model, input_data)
    elif model_type == 'brain_tumor':
        return predict_brain_tumor_image(model, input_data)

# === Flask Application ===

app = Flask(__name__)

# Serve the homepage
@app.route('/')
def home():
    return render_template('homepage.html')

# Serve the retinopathy detection page
@app.route('/detect/retinopathy')
def detect_retinopathy():
    return render_template('retinopathy.html')

# Serve the brain tumor detection page
@app.route('/detect/brain-tumor')
def detect_brain_tumor():
    return render_template('tumour.html')

# Handle retinopathy prediction
@app.route('/predict/dr', methods=['POST'])
def predict_dr():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    
    result = predict('dr', temp_path)
    
    os.remove(temp_path)
    
    return jsonify({'prediction': result})

# Handle brain tumor prediction
@app.route('/predict/brain_tumor', methods=['POST'])
def predict_brain():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    
    result = predict('brain_tumor', temp_path)
    
    os.remove(temp_path)
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
