import io
import requests
import numpy as np
import onnxruntime
from PIL import Image
from model import  Model , Preprocessor
from torchvision import transforms


# Load your ONNX model as a global variable here using the variable name "model"
def init():
    global model

    model_file = "pytorch_weights.onnx"
    model = onnxruntime.InferenceSession(model_file)
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.

def inference(model_inputs):
    # Convert the image data to a PIL Image object
    image = Image.open(io.BytesIO(model_inputs['input']))
    
    # Apply the image transform and convert to a numpy array
    resize = transforms.Resize((224, 224))
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    image = resize(image)
    image = crop(image)
    image = to_tensor(image)
    image = normalize(image)
    

    # Use the ONNX model to make a prediction
    input_name = model.get_inputs()[0].name
    #output_name = model.get_outputs()[0].name
    output = model.run(None, {input_name: image})[0]
    predicted_class = np.argmax(output)

    # Convert the prediction to a JSON response
    response = {"class": str(predicted_class)}

    return predicted_class

    
