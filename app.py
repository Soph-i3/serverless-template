from model import Model, Preprocessor

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
# def init():
#     global model
    
#     device = 0 if torch.cuda.is_available() else -1
#     model = pipeline('fill-mask', model='bert-base-uncased', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(image_file, image_bytes, onnx_file):
    p = Preprocessor(image_bytes)
    pimg = p.preprocess_numpy(p.load_image())
    pimg = tuple([image_file, pimg[1]])
    
    m = Model(pimg, onnx_file)

    # Return the results as a dictionary
    return {"label": m.load_and_predict()[0], "class": str(m.load_and_predict()[1])}