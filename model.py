# load onnx model
import onnxruntime as onnxrt
from PIL import Image
from torchvision import transforms
from pytorch_model import Classifier, BasicBlock
import numpy as np


class Model:
    def __init__(self, image_data, onnx_file):
        self.image_file = image_data[0]
        self.preprocessed_image = image_data[1]
        self.onnx_file = onnx_file
        
        
    def load_and_predict(self):
        # load dependency
        obj = Classifier(BasicBlock, [2, 2, 2, 2])

        # create session for the onnx model
        sess = onnxrt.InferenceSession(self.onnx_file)
        
        # run session
        out = sess.run(None, {"input": self.preprocessed_image.unsqueeze(0).numpy()})

        return self.image_file.split(".")[0], np.argmax(out)
    
    
class Preprocessor:
    def __init__(self, image_file):
        self.image_file = image_file
        
    def load_image(self):
        return Image.open(self.image_file)
    
    def preprocess_numpy(self, img):
        resize = transforms.Resize((224, 224))   #must same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        
        return self.image_file, img 