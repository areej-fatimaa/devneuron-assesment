from fastapi import FastAPI, UploadFile, File, Form
import torch
from PIL import Image
import io
import base64
from fgsm import FGSM
from model import load_model
from utils import preprocess_image
import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://main.d3abcxyz.amplifyapp.com"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
attack = FGSM(model)

@app.post("/attack")
async def run_attack(
    file: UploadFile = File(...),
    epsilon: float = Form(0.1)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    tensor = preprocess_image(image)

    # Clean prediction
    output = model(tensor)
    clean_pred = output.argmax(dim=1).item()

    label = torch.tensor([clean_pred])

    # FGSM attack
    adv_tensor = attack.attack(tensor, label, epsilon)

    adv_output = model(adv_tensor)
    adv_pred = adv_output.argmax(dim=1).item()

    success = clean_pred != adv_pred

    # Convert adversarial image to base64
    adv_img = adv_tensor.squeeze().detach().numpy() * 255
    adv_img = Image.fromarray(adv_img.astype("uint8"))

    buffer = io.BytesIO()
    adv_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    with open("outputs/fgsm_results.txt", "a") as f:
     f.write("=================================\n")
     f.write(f"Time: {datetime.datetime.now()}\n")
     f.write(f"Epsilon: {epsilon}\n")
     f.write(f"Clean Prediction: {clean_pred}\n")
     f.write(f"Adversarial Prediction: {adv_pred}\n")
     f.write(f"Attack Success: {success}\n\n")
    return {
        "clean_prediction": clean_pred,
        "adversarial_prediction": adv_pred,
        "attack_success": success,
        "adversarial_image": img_base64
    }
@app.get("/health")
def health():
    return {"status": "ok"}
