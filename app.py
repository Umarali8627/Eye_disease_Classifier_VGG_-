from flask import Flask, request, jsonify , render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from io import BytesIO

# load the model 
model=tf.keras.models.load_model('eye_disease_model_VGG19.h5')
# allow extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Define the class names
class_names=['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # ...existing code...
    if file and allowed_file(file.filename):
      try:
        # Read the image file using BytesIO
        
        img = tf.keras.preprocessing.image.load_img(BytesIO(file.read()), target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create batch axis

        # Make prediction
        predictions = model.predict(img_array)
        idx = np.argmax(predictions[0])
        pred_class = class_names[idx]
        confidence = float(np.max(predictions[0]))
        

        return jsonify({
            'predicted_class': pred_class,
            'confidence': confidence  
        })
      except Exception as e:
        return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Invalid file type'}), 400
if __name__ == '__main__':
    app.run(debug=True)
        