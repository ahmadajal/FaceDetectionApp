import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType

face_client = FaceClient("https://gettingstartedface.cognitiveservices.azure.com/",
                         CognitiveServicesCredentials("2c1bc8c024534cceb8fe0553ee92636b"))

def face_detect(url):
    # Detect a face in an image that contains a single face
    single_face_image_url = url
    single_image_name = os.path.basename(single_face_image_url)
    detected_faces = face_client.face.detect_with_url(url=single_face_image_url,
                                                      retrun_face_landmarks=True,
                                                     attributes='small,age')
    if not detected_faces:
        raise Exception('No face detected from image {}'.format(single_image_name))

    # Convert width height to a point in a rectangle
    def getRectangle(faceDictionary):
        rect = faceDictionary.face_rectangle
        left = rect.left
        top = rect.top
        right = left + rect.width
        bottom = top + rect.height

        return ((left, top), (right, bottom))


    # Download the image from the url
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

    # For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red', width=1)

    return img
