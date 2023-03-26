import io
import requests
import numpy as np
import onnxruntime
from PIL import Image
from model import  Preprocessor
from torchvision import transforms


# Load your ONNX model as a global variable here using the variable name "model"
def init():
    global model

    model_file = "pytorch_weights.onnx"
    model = onnxruntime.InferenceSession(model_file)
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.

def inference(model_inputs):
    # Check if model_inputs is a dictionary
    if not isinstance(model_inputs, dict):
        return "incorrect"

    # Get the image data from the link
    image_url = model_inputs['input']
    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)

    # Convert the image data to a PIL Image object
    image = Image.open(image_bytes).convert('RGB')

    # Apply the image transform and convert to a numpy array
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).numpy()

    # Use the ONNX model to make a prediction
    input_name = model.get_inputs()[0].name
    #output_name = model.get_outputs()[0].name
    output = model.run(None, {input_name: image})[0]
    predicted_class = np.argmax(output)

    # Convert the prediction to a JSON response
    response = {"class": str(predicted_class)}

    return predicted_class
# def inference(model_inputs):
#     # Check if model_inputs is a dictionary
#     if not isinstance(model_inputs, dict):
#         return "incorrect"

#     # Convert the image data to a PIL Image object
#     image = Image.open(io.BytesIO(model_inputs['image_data']))
    
#     # Apply the image transform and convert to a numpy array
#     image = Preprocessor.preprocess_numpy(image).unsqueeze(0).numpy()

#     # Use the ONNX model to make a prediction
#     input_name = model.get_inputs()[0].name
#     #output_name = model.get_outputs()[0].name
#     output = model.run(None, {input_name: image})[0]
#     predicted_class = np.argmax(output)

#     # Convert the prediction to a JSON response
#     response = {"class": str(predicted_class)}

#     return predicted_class

    
