from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageForm
import tensorflow as tf
from keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import os
import cv2
import pandas as pd

Flower_model = tf.lite.Interpreter(os.path.join('models', 'Flower_lite_model.tflite'))
labels_name = pd.read_csv(os.path.join('models', 'label_names.csv'))
labels_name = dict(labels_name.values)

def classify_view(request):
    if request.method == 'POST':
        imgForm = ImageForm(request.POST, request.FILES)
        if imgForm.is_valid():
            img = imgForm.save()
            img_path = img.image.url
            path = os.path.join('media',str(img.image))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32')
            img = preprocess_input(img)
            Flower_model.allocate_tensors()
            index_input = Flower_model.get_input_details()[0]['index']
            Flower_model.set_tensor(index_input, img[None,...])
            Flower_model.invoke()
            index_output = Flower_model.get_output_details()[0]['index']
            lbl = Flower_model.get_tensor(index_output)
            percent_lbl = str(tf.reduce_max(lbl).numpy() * 100)[:5]
            lbl = tf.argmax(lbl, axis=-1)
            lbl_name = labels_name[lbl.numpy()[0]]

            return render(request, 'index.html', {'form': imgForm, 'image_path': img_path, 'predicted_label': lbl_name, 'percent_label': percent_lbl})


            
    imgForm = ImageForm()
    return render(request, 'index.html', {'form':imgForm})
