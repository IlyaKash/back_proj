from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from models import models, utils_resnet_TL, postproc_2
import gdown
import os

app = FastAPI()

# Параметры модели
img_size = 224
num_classes = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# Загрузка моделей
change_net = models.ChangeNet(num_classes=num_classes).to(device)
model_potato = models.get_model().to(device)
model_onion = models.get_model().to(device)
model_cabbage = models.get_model().to(device)

# Загрузка весов моделей
change_net.load_state_dict(torch.load('classificators/best_model_new_3.pkl', map_location=device))
model_potato.load_state_dict(torch.load('classificators/efficientnet_v2S_potato_l1280_l32_0.9675_16_16_20_05.pt', map_location=device))
model_onion.load_state_dict(torch.load('classificators/efficientnet_v2s_onion_l1280_lrelu_l320.935_07_25_21_05.pt', map_location=device))
model_cabbage.load_state_dict(torch.load('classificators/efficientnet_v2s_cabbage_l1280_lrelu_l320.9725_13_16_23_05.pt', map_location=device))

change_net.eval()
model_potato.eval()
model_onion.eval()
model_cabbage.eval()

# Трансформации изображений
trf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trf_binary = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Маршрут для предсказания
@app.post("/predict/")
async def predict(ref_img: UploadFile = File(...), test_img: UploadFile = File(...)):
    ref_img_orig = Image.open(BytesIO(await ref_img.read())).convert('RGB')
    test_img_orig = Image.open(BytesIO(await test_img.read())).convert('RGB')

    ref_img_transformed = trf(ref_img_orig).unsqueeze(0).to(device)
    test_img_transformed = trf(test_img_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = change_net([ref_img_transformed, test_img_transformed])
        _, output = torch.max(pred, 1)

    mask = output.squeeze(0).cpu().numpy()
    t1 = np.array(test_img_orig)
    t1 = cv2.cvtColor(t1, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(t1, (img_size, img_size), interpolation=cv2.INTER_AREA)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype('uint8')
    dilate_iter = 10
    mask = cv2.dilate(mask.astype('uint8'), kernel, iterations=dilate_iter)
    resized[mask == 0] = 1

    with torch.no_grad():
        potato = int(model_potato(trf_binary(torch.tensor(resized).permute(2, 0, 1).to(device)).unsqueeze(0))[0].round())
        onion = int(model_onion(trf_binary(torch.tensor(resized).permute(2, 0, 1).to(device)).unsqueeze(0))[0].round())
        cabbage = int(model_cabbage(trf_binary(torch.tensor(resized).permute(2, 0, 1).to(device)).unsqueeze(0))[0].round())

    # Визуализация результатов
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 2, 1)
    plt.imshow(ref_img_orig)
    plt.title('Reference Image')

    fig.add_subplot(2, 2, 2)
    plt.imshow(test_img_orig)
    plt.title('Test Image')

    fig.add_subplot(2, 2, 3)
    plt.text(0, -30, f'Potato: {potato}, Onion: {onion}, Cabbage: {cabbage}', ha='left')
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title('Change Detection Output')

    fig.add_subplot(2, 2, 4)
    plt.imshow(mask)
    plt.title('ChangeNet Mask')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
