import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

from model import load_model
from utils import preprocess_image
from gradcam import generate_gradcam
from feature_maps import feature_maps_to_images

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://medicinal-plant-identification-brown.vercel.app"],
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = torch.load("models/classes.pth")

model = load_model("models/densenet121_best.pth", len(classes), device)

total_params = sum(p.numel() for p in model.parameters())

@app.get("/")
def home():
    return {"message": "Plant Classification API 🌿"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        image = Image.open(file.file).convert("RGB")
        input_tensor = preprocess_image(image)

        with torch.no_grad():
            outputs = model(input_tensor.to(device))
            probs = F.softmax(outputs, dim=1)

        k = min(3, probs.shape[1])
        top_probs, top_idxs = torch.topk(probs, k)

        predictions = [
            {
                "species": classes[top_idxs[0][i].item()],
                "confidence": round(top_probs[0][i].item(), 4)
            }
            for i in range(k)
        ]

        # Feature maps and Grad-CAM
        grad_results = generate_gradcam(model, input_tensor, device)
        heatmap = grad_results["heatmap"]
        feature_maps = feature_maps_to_images(grad_results["activations"], max_maps=8)

        inference_time = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "data": {
                "predictions": predictions,
                "featureMaps": feature_maps,
                "attentionHeatmap": heatmap,
                "processingStats": {
                    "modelName": "DenseNet121",
                    "inferenceTime": f"{inference_time}ms",
                    "confidence": round(top_probs[0][0].item(), 2),
                    "layers": 121,
                    "parameters": f"{round(total_params / 1e6, 2)}M"
                }
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)