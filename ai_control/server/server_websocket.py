import asyncio
import websockets
import numpy as np
import base64
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import PIL
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Labels = {
    0: "Attack",
    1: "Bottom",
    2: "Left",
    3: "Right",
    4: "Stop",
    5: "Top",
}

def pretrain_model(name_model='mobi-v2'):
    pretrain_model = models.mobilenet_v2()
    pretrain_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 6))
    path_model_pretrain = "./weight/hand_model_mobi_v2.pt"

    if name_model == 'mobi-v3-l':
        pretrain_model = models.mobilenet_v3_large()
        pretrain_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 6))
        path_model_pretrain = "./weight/hand_model_mobi_v3_large.pt"
    if name_model == 'mobi-v3-s':
        pretrain_model = models.mobilenet_v3_small()
        pretrain_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 6))
        path_model_pretrain = "./weight/hand_model_mobi_v3_small.pt"

    print(path_model_pretrain)

    pretrain_model.to(device)
    pretrain_model.load_state_dict(
        torch.load(path_model_pretrain, map_location=device), strict=False
    )
    pretrain_model.eval()

    return pretrain_model

transformImage = transforms.Compose([
                                      transforms.Resize((240, 240)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5345, 0.5550, 0.5419],
                                                           [0.2360, 0.2502, 0.2615])
                                    ])
def transform_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize([0.5345, 0.5550, 0.5419], 
                                 [0.2360, 0.2502, 0.2615]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def classificationTest(model, image):
    image = PIL.Image.fromarray(image)
    image = transformImage(image).float()
    image = image.unsqueeze(0)
    image.to(device)
    out = model(image)

    _, pre = torch.max(out.data, 1)
    prob = F.softmax(out, dim=1)
    top_p, top_class = prob.topk(1, dim = 1)
    return Labels[pre.item()], top_p.item()

def classificationApi(model, image):
    image = transform_image(image)
    image.to(device)
    out = model(image)

    _, pre = torch.max(out.data, 1)
    prob = F.softmax(out, dim=1)
    top_p, top_class = prob.topk(1, dim = 1)
    return Labels[pre.item()], top_p.item()

def chuyen_base64_sang_anh(anh_base64):
    try:
        anh_base64 = np.frombuffer(
            base64.b64decode(anh_base64), dtype=np.uint8)
    except:
        return None
    return anh_base64

connected = set()

async def server(websocket, path):
    # Register.
    connected.add(websocket)
    print("ws", websocket)
    try:
        async for message in websocket:
            for conn in connected:
                if conn == websocket:
                    try:
                        imageFile = message.split(',')[1]
                        image = chuyen_base64_sang_anh(imageFile)
                        model = pretrain_model('mobi-v2')
                        prediction, percent = classificationApi(model, image)
                        data = str({'Class Name': prediction, 'Percent': percent*100})
                        await conn.send(data)
                        print(data)

                    except Exception as e:
                        error = "'Error': '" + str(e) + "'"
                        conn.send(error)
                        print(error)
    finally:
        # Unregister.
        connected.remove(websocket)
    

start_server = websockets.serve(server, "0.0.0.0", 5001)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()