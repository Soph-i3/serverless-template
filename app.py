import json
import base64
import io
import numpy as np
import onnxruntime
from PIL import Image
from model import  Model, Preprocessor

# Load your ONNX model as a global variable here using the variable name "model"
def init():
    global model

    model_file = "/pytorch_weights.onnx"
    model = onnxruntime.InferenceSession(model_file)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs):
    input_image = model_inputs.get('image', None)
    if input_image is None:
        return {'message': "No image provided"}
    
    p = Preprocessor()
    img = p.load_image(input_image)
    pimg = img.preprocess_numpy(img)
    
    
    # Run the ONNX model on the image
    
    result = model.run(None,{"input": pimg.unsqueeze(0).numpy()})
    
    # Extract the relevant output values from the result dictionary
    class_idx = np.argmax(result)
#     class_name = "class name corresponding to class_idx"
#     score = float(result[class_idx])
    
    # Return the results as a dictionary
    return {'class_idx': class_idx}