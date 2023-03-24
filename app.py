import json
import base64
import io
import numpy as np
import onnxruntime
from PIL import Image
from model import Preprocessor

# Load your ONNX model as a global variable here using the variable name "model"
def init():
    global model

    model_file = "/pytorch_weights.onnx"
    model = onnxruntime.InferenceSession(model_file)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(payload):
    data = json.loads(payload['body'])
    image_data = data['image']
    
    # Decode image data and preprocess the image
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    img = Preprocessor.preprocess_numpy(image_data)
    # Run the ONNX model on the image
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: img})[0]
    
    # Extract the relevant output values from the result dictionary
    class_idx = np.argmax(result)
#     class_name = "class name corresponding to class_idx"
#     score = float(result[class_idx])
    
    # Return the results as a dictionary
    return {'class_idx': class_idx}