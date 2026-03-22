from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch

from model import load_model
from utils import preprocess_image

app = FastAPI()

# Load classes
classes = torch.load("models/classes.pth")

# Load model
model = load_model("models/densenet121_best.pth", num_classes=len(classes))


@app.get("/")
def home():
    return {"message": "Plant Classification API 🌿"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted.item()]

    return {
        "prediction": predicted_class
    }