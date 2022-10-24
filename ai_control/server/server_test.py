from re import I
from flask import Flask, redirect, url_for
from flask_cors import CORS, cross_origin
from flask import request
from flask import render_template, jsonify
import numpy as np
import base64
import requests
from PIL import Image
from io import BytesIO
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
import cv2

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
    path_model_pretrain = "../weight/hand_model_mobi_v2.pt"

    if name_model == 'mobi-v3-l':
        pretrain_model = models.mobilenet_v3_large()
        pretrain_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 6))
        path_model_pretrain = "../weight/hand_model_mobi_v3_large.pt"
    if name_model == 'mobi-v3-s':
        pretrain_model = models.mobilenet_v3_small()
        pretrain_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 6))
        path_model_pretrain = "../weight/hand_model_mobi_v3_small.pt"

    print(path_model_pretrain)

    pretrain_model.to(device)
    pretrain_model.load_state_dict(
        torch.load(path_model_pretrain, map_location=device), strict=False
    )
    pretrain_model.eval()

    return pretrain_model

# def transform_image(image_bytes):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((240, 240)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5345, 0.5550, 0.5419],
#                                 [0.2360, 0.2502, 0.2615])
#         ]
#     )
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
#     return transform(image).unsqueeze(0)

# def get_prediction(name_model, image):
#     image.to(device)
#     output = (pretrain_model(name_model))(image)

#     _, predictions = torch.max(output.data, 1)
#     prob = F.softmax(output, dim=1)
#     top_p, _ = prob.topk(1, dim=1)
#     return Labels[predictions.item()], top_p.item()

test_transform = transforms.Compose([#SquarePad(),
                                      transforms.Resize((240, 240)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5345, 0.5550, 0.5419],
                                                           [0.2360, 0.2502, 0.2615])
                                    ])

# def test_transform(image):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((240, 240)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5345, 0.5550, 0.5419],
#                                 [0.2360, 0.2502, 0.2615])
#         ]
#     )
#     # image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
#     return transform(image).unsqueeze(0)

def classification(model, image):
    image = PIL.Image.fromarray(image)
    image = test_transform(image).float()
    image = image.unsqueeze(0)
    image.to(device)
    #out = pretrain_model(image)
    out = model(image)
    
    _, pre = torch.max(out.data, 1)
    
#     sm = torch.nn.Softmax()
#     sm.eval()
#     probabilities = sm(out) 
# #     print(probabilities) #Converted to probabilities
#     probabilities = probabilities.detach().numpy()
#     print("MAX : " + str(max(probabilities)))
    
    prob = F.softmax(out, dim=1)
    
    
    top_p, top_class = prob.topk(1, dim = 1)
    #print(top_p)
    return Labels[pre.item()], top_p.item()
                                  
app = Flask(__name__)

@app.route('/test/<name_model>', methods=['GET', 'POST'])
@cross_origin(origin='*')
def mainpage(name_model):   
    if request.method == 'POST':
        image = request.files['imagefile']  # get file
        image_b64 = base64.b64encode(image.read()).decode('utf-8')
        image_save=Image.open(BytesIO(base64.b64decode(image_b64)))
        image_save.save("term.png")
        # image = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
        image_save = np.array(image_save)
        # print("abc:", image_save.shape)
        # print("xyz:", image.shape)
        # img_resize = cv2.resize(image, (240, 240))
        # tensor = test_transform(image_save)
        print(type(image_save))
        print(image_save.shape)
        model = pretrain_model()
        prediction, percent = classification(model, image_save)
        return render_template('index.html', name_model=name_model, prediction=prediction, percent=percent) 
    return render_template('index.html', name_model=name_model)

# @app.route('/test/<name_model>', methods=['GET', 'POST'])
# @cross_origin(origin='*')
# def mainpage(name_model):   
#     if request.method == 'POST':
#         image = request.files['imagefile']  # get file
#         image_b64 = base64.b64encode(image.read()).decode('utf-8')
#         image_save=Image.open(BytesIO(base64.b64decode(image_b64)))
#         image_save.save("term.png")
#         image = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
#         tensor = transform_image(image)
#         prediction, percent = get_prediction(name_model, tensor)
#         return render_template('index.html', name_model=name_model, prediction=prediction, percent=percent) 
#     return render_template('index.html', name_model=name_model)

# @app.route('/api/<name_model>', methods=['GET', 'POST'])
# @cross_origin(origin='*')
# def apiProcess(name_model):   
#     if request.method == 'POST':
#         try:
#             image = request.files['imageFile']
#             image_b64 = base64.b64encode(image.read()).decode('utf-8')
#             image = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
#             tensor = transform_image(image)
#             prediction, percent = get_prediction(name_model, tensor)
#             data = {'Class Name': prediction, 'Percent': percent*100}
#             print(prediction)
#             return jsonify(data)
#         except Exception as e:
#             error = "'Error': '" + str(e) + "'"
#             print(error)
#             return jsonify({'Error': str(e)})

@app.route('/')
@cross_origin(origin='*')
def init():
    return redirect('test/mobi-v2')

if __name__ == "__main__":
    app.run(debug=True)