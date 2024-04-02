import torch
from fastapi import FastAPI
from process_data import get_tokens
from utils import Predictor

server = FastAPI(title="pytorch-gpt-wikihow")
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {} for API inference".format(compute_device))
data_tensors_path = "data_tensors"
model_path = "deployment/model.pt"

predictor = Predictor(model_path, data_tensors_path, compute_device)


@server.get("/predict")
async def predict(prompt: str, num_tokens: int, temperature: float = 1.0):
    output = predictor.predict_tokens(get_tokens(prompt), num_tokens, temperature)
    output = Predictor.beautify_output(output)
    return {"response": output}
