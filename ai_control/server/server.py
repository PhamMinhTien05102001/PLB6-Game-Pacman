from flask import Flask, redirect
from flask_cors import cross_origin
from flask import request
from flask import render_template, jsonify
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import PIL

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

transformImage = transforms.Compose([
                                      transforms.Resize((240, 240)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5345, 0.5550, 0.5419],
                                                           [0.2360, 0.2502, 0.2615])
                                    ])

def classification(model, image):
    image = PIL.Image.fromarray(image)
    image = transformImage(image).float()
    image = image.unsqueeze(0)
    image.to(device)
    out = model(image)

    _, pre = torch.max(out.data, 1)
    prob = F.softmax(out, dim=1)
    top_p, top_class = prob.topk(1, dim = 1)
    return Labels[pre.item()], top_p.item()

app = Flask(__name__)

@app.route('/test/<name_model>', methods=['GET', 'POST'])
@cross_origin(origin='*')
def mainpage(name_model):   
    if request.method == 'POST':
        imageFile = request.files['imagefile']  # get file
        image_b64 = base64.b64encode(imageFile.read()).decode('utf-8')
        image=Image.open(BytesIO(base64.b64decode(image_b64)))
        image = np.array(image)

        model = pretrain_model(name_model)
        prediction, percent = classification(model, image)

        return render_template('index.html', name_model=name_model, prediction=prediction, percent=percent) 
    return render_template('index.html', name_model=name_model)

@app.route('/api/<name_model>', methods=['GET', 'POST'])
@cross_origin(origin='*')
def apiProcess(name_model):   
    if request.method == 'POST':
        try:
            imageFile = request.files['imageFile']  # get file
            image_b64 = base64.b64encode(imageFile.read()).decode('utf-8')
            image=Image.open(BytesIO(base64.b64decode(image_b64)))
            image = np.array(image)

            model = pretrain_model(name_model)
            prediction, percent = classification(model, image)
            data = {'Class Name': prediction, 'Percent': percent*100}
            print(data)
            return jsonify(data)
        except Exception as e:
            error = "'Error': '" + str(e) + "'"
            print(error)
            return jsonify({'Error': str(e)})

@app.route('/')
@cross_origin(origin='*')
def init():
    return redirect('test/mobi-v2')

if __name__ == "__main__":
    app.run(debug=True)