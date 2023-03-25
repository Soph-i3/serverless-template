import json
import base64
import io
import numpy as np
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
from model import  Model, Preprocessor


# Load your ONNX model as a global variable here using the variable name "model"
def init():
    global model

    model_file = "pytorch_weights.onnx"
    model = onnxruntime.InferenceSession(model_file)
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs):

    # Convert the image data to a PIL Image object
    image = Image.open(io.BytesIO(model_inputs))
    
    # Apply the image transform and convert to a numpy array
    image = Preprocessor.preprocess_numpy(image).unsqueeze(0).numpy()

    # Use the ONNX model to make a prediction
    input_name = model.get_inputs()[0].name
    #output_name = model.get_outputs()[0].name
    output = model.run(None, {input_name: image})[0]
    predicted_class = np.argmax(output)

    # Convert the prediction to a JSON response
    response = {"class": str(predicted_class)}

    return response
    
#     input_image = model_inputs['image']
    
#     p = Preprocessor()
#     img = p.load_image(input_image)
#     pimg = img.preprocess_numpy(img)
    
    
#     # Run the ONNX model on the image
    
#     result = model.run(None,{"input": pimg.unsqueeze(0).numpy()})
    
#     # Extract the relevant output values from the result dictionary
#     class_idx = np.argmax(result)
# #     class_name = "class name corresponding to class_idx"
# #     score = float(result[class_idx])
    
#     # Return the results as a dictionary
#     return {'class_idx': class_idx}
