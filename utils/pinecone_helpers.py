import pinecone
from preprocess import preprocess_image
from model.load_model import load_model
import torch

# Initialize Pinecone
pc = pinecone(api_key="pcsk_2agnRW_PJGcuYNduPh4LXPttu1bnT31UninNBGbyZqKDUiCDAxdY7asgDJkLHmR3KCDJzP")
index = pc.Index("new")
model, _ = load_model()

def query_pinecone(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        query_embedding = model.encode_image(image_tensor).cpu().numpy().flatten()
    
    response = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    return [{"id": match["id"], "url": match["metadata"]["url"], "score": match["score"]} for match in response["matches"]]
