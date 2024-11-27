import torch
import open_clip
from PIL import Image

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the specified model and transforms for training and validation
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
    )
    
    # Load the tokenizer
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    
    # Move the model to the appropriate device
    model = model.to(device)
    
    return model, preprocess_train

# # Output the loaded components
# model, preprocess_train = load_model()
# print("Model:", model)
# print("Preprocessing for training:", preprocess_train)
