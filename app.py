from flask import Flask, render_template, request
from predict import predict_image

app = Flask(__name__)

# Categories of diseases
categories = ['eczema', 'melanoma', 'atopic_dermatitis', 'basal_cell_carcinoma', 'melanotic_nevi']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']  # This is a FileStorage object
    
    if file.filename == '':
        return "No file selected", 400
    
    result = predict_image("skin_disease_model.h5", file, categories=categories)
    return render_template('detect.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
