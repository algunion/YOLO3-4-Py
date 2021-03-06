from pydarknet import Detector, Image
import cv2
import os
import flask
import numpy as np
import flask
import io
from PIL import Image as pil
from base64 import b64encode, b64decode, decodestring
from io import BytesIO
import json
from time import time

app = flask.Flask(__name__)
model = None
graph = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))


@app.route("/objects", methods=["GET"])   
def objects():
    return flask.jsonify(model.class_names)

    
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":       
        
        decoded = flask.request.data.decode("utf-8")
        print ("decoding ready")
        received =  json.loads(decoded)
        
        if received.get("image"):
            # read the image in PIL format            
            start_img_proc = time()
            image = b64decode(received["image"])
            image = pil.open(io.BytesIO(image)) 
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            darknet_img = Image(cv2_image) 
            img_proc = time() - start_img_proc
            print("Img proc time:")
            print(img_proc)

            model_start = time()
            results = model.detect(darknet_img)
            model_proc = time() - model_start
            print("model proc time")
            print(model_proc)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(cv2_image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
                cv2.putText(cv2_image,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_DUPLEX,4,(0,0,255), thickness=2)                
    

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = b64encode(buffered.getvalue())
            img_str = img_bytes.decode('utf-8')
            return flask.jsonify({"result": img_str})
        
    return flask.jsonify({"failure": "Not valid input", "received": str(flask.request.data)})
    
@app.route("/objdetect", methods=["POST"])
def objdetect():
    # initialize the data dictionary that will be returned from the
    # view
    decoded = flask.request.data.decode("utf-8")
    print ("decoding ready")
    received =  json.loads(decoded)

    # ensure an image was properly uploaded to our endpoint
    if received.get("image"):
        # read the image in PIL format            
        image = b64decode(received["image"])
        image = pil.open(io.BytesIO(image)) 
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        darknet_img = Image(cv2_image) 

        result = []
        
        results = model.detect(darknet_img)    

        for cat, score, bounds in results:
            detected_class = cat.decode('utf-8')
            x, y, w, h = bounds
            result.append({"class": detected_class, "score": float(score), "box": { "x": int(x), "y": int(y), "w":int(w), "h": int(h)}})

        
        print(result)
        
        return flask.jsonify(result)
        
    return flask.jsonify({"failure": "Not valid input", "received": str(flask.request.data)})
    
    
    
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host= '0.0.0.0')