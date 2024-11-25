from PIL import Image
from model.model_load import load_model

model, preprocess = load_model()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)