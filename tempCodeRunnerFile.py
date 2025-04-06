from flask import Flask, render_template, request, jsonify, send_file
from fpdf import FPDF
import os
import datetime
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# === Initialize Flask App ===
app = Flask(__name__, static_folder='static', template_folder='templates')

# === Global Variables ===
latest_result = {}  # Stores the latest diagnosis result
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

# Try to load models, but handle if files don't exist
try:
    dr_model = load_dr_model(MODEL_PATHS['dr'])
    brain_tumor_model = load_brain_tumor_model(MODEL_PATHS['brain_tumor'])
except FileNotFoundError:
    print("Warning: Model files not found. Prediction functionality will be limited.")
    dr_model = None
    brain_tumor_model = None

def predict(model_type, input_data):
    if model_type == 'dr' and dr_model:
        return predict_dr_image(dr_model, input_data)
    elif model_type == 'brain_tumor' and brain_tumor_model:
        return predict_brain_tumor_image(brain_tumor_model, input_data)
    else:
        return "Model not available"

# === Prediction Functions ===
def predict_dr_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return classes[predicted.item()]

# === Define simplified chatbot function if import fails ===
try:
    from chatbot import chatbot_response
except ImportError:
    def chatbot_response(message):
        return "I'm a medical assistant that can help answer your questions about diagnoses and treatments."

# === Flask Routes ===
@app.route('/')
def home():
    return render_template('homepage.html')  # Use index.html as homepage

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/consult')
def consult():
    return render_template('consult.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/video/token', methods=['POST'])
def video_token():
    # This would integrate with a service like Twilio, Agora, etc.
    # For example with Twilio:
    # identity = request.json.get('identity', '')
    # room = request.json.get('room', '')
    # token = generate_video_token(identity, room)  # You'd implement this function
    # return jsonify({'token': token})
    
    # For now, return a placeholder response
    return jsonify({'message': 'Video token endpoint - implementation needed'})

@app.route('/save_consultation', methods=['POST'])
def save_consultation():
    # In a real implementation, you would:
    # 1. Validate the data
    # 2. Store consultation records
    # 3. Link it with the patient's diagnosis data
    
    data = request.json
    patient_name = data.get('patient_name')
    doctor_name = data.get('doctor_name')
    consultation_type = data.get('type')  # 'chat' or 'video'
    notes = data.get('notes', '')
    
    # For now, just return success
    return jsonify({
        'success': True, 
        'message': 'Consultation record saved',
        'consultation_id': 'CONS' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    })

@app.route('/get_latest_result')
def get_latest_result():
    # Add recommended specialist based on diagnosis
    if latest_result.get('result') in ['Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
        latest_result['recommended_specialist'] = 'Ophthalmologist'
    elif latest_result.get('result') in ['glioma', 'meningioma', 'pituitary']:
        latest_result['recommended_specialist'] = 'Neurologist'
    
    return jsonify(latest_result)

@app.route('/chat')
def chat_page():
    return render_template('chat.html') 

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'response': 'Please enter a message.'})

    bot_response = chatbot_response(user_message)  # Get response from chatbot
    return jsonify({'response': bot_response})
    
@app.route('/download_report')
def download_report():
    patient_name = request.args.get('name', 'Anonymous')
    if not latest_result:
        return jsonify({'error': 'No diagnosis available'}), 400

    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Details
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
    
    # Diagnosis Result
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, txt=f"Diagnosis: {latest_result['result']}", ln=True, align='L')
    pdf.ln(10)
    
    # Detailed Information and Precautions
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="This report provides a preliminary diagnosis based on deep learning analysis. "
                                "Consult a medical professional for comprehensive evaluation and treatment.", align='L')
    
    # Disclaimer
    pdf.set_font("Arial", style='I', size=10)
    pdf.cell(0, 10, txt="Generated by AI Diagnosis System - Not a Substitute for Professional Medical Advice", ln=True, align='C')
    
    # Save PDF
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", f"{patient_name}_diagnosis_report.pdf")
    pdf.output(report_path)
    
    return send_file(report_path, as_attachment=True, download_name=f"{patient_name}_diagnosis_report.pdf")

@app.route('/predict/dr', methods=['POST'])
def predict_dr():
    global latest_result
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    try:
        result = predict('dr', temp_path)
        latest_result = {'result': result}
        return jsonify({'prediction': result})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/predict/brain_tumor', methods=['POST'])
def predict_brain():
    global latest_result
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    temp_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(temp_path)
    try:
        result = predict('brain_tumor', temp_path)
        latest_result = {'result': result}
        return jsonify({'prediction': result})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
@app.route('/signup', methods=['POST'])
def signup():
    # Get form data
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    terms_accepted = data.get('terms_accepted')
    
    # In a real application, you would:
    # 1. Validate the data
    # 2. Check if email already exists
    # 3. Hash the password
    # 4. Store user in database
    
    # For now, just return success (you would implement actual user registration later)
    return jsonify({'success': True, 'message': 'User registered successfully'})

if __name__ == '__main__':
    app.run(debug=True)