
# coding: utf-8

# In[69]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from time import time
import os
import pypianoroll as pr
import numpy as np
import random
from random import shuffle


# In[2]:


import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

tf.keras.backend.set_session(sess)


# In[13]:


def load_model():
    # load JSON and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return loaded_model
loaded_model = load_model()


# In[3]:


#load midi files with melody tracks
mel_roll = []
def load_mel_midi():
     for root, dirs, files in os.walk("./midi_with_mel/", topdown=False):
        for name in files:
            full_name = os.path.join(root,name)
            temp = pr.parse(full_name)
            temp.name = name.split('.')[0]
            mel_roll.append(temp)
load_mel_midi()


# In[38]:


#handle the prediction returned from the model
def handle_prediction(prediction,random,length):
    label = []
    if random != 0:
        prediction = prediction.reshape(int(prediction.shape[0]/random),random,1)
    elif random == 0:
        prediction = prediction.reshape(int(prediction.shape[0]/length),length,1)
    avg_score = np.average(prediction,axis=1)
    max_index = np.argmax(avg_score,axis = 0)
#     for x in range(len(song.tracks)):
#         if avg_score[x]:
#             song.tracks[x].name = "Melody"
    
    return max_index


# In[73]:


def predict_song(song,loaded_model,random):
    img_rows = 128
    img_cols = 500
    store = []
    temp = []
    temp_song = song
    #iterate over tracks and randomly sample 3 subarrays in a track if random is set
    for track in temp_song.tracks:
        if random != 0:
            random_ints = random.sample(range(0,track.pianoroll.shape[0]-500),random)
            for x in range(0,random):
                temp_array = np.array(track.pianoroll[random_ints[x]:random_ints[x]+500,:])
                temp.append((temp_array).swapaxes(0,1))
        elif random == 0:
            for x in range(0,int(track.pianoroll.shape[0]/500)):
                temp_array = np.array(track.pianoroll[500*x:500*(x+1),:])
                temp.append((temp_array).swapaxes(0,1))
            length = len(temp)
        temp = np.array(temp)
        temp = temp.reshape(temp.shape[0],img_rows,img_cols,1)
        store.extend(temp)
        temp = []
    store = np.array(store)
    prediction = loaded_model.predict(store,verbose=1)
    max_index = handle_prediction(prediction,random,length)
    return max_index


# In[74]:


song_count = 0
success = 0
mel_track = []
for song in mel_roll:
    print(song.name)
    if song_count == 2:
        break
    else:
        tracks = song.tracks
        shuffle(tracks)
        for x in range(len(tracks)):
#             print(tracks[x].name)
            if tracks[x].name.lower() == 'melody':
                mel_track.append(x)
        max_index = predict_song(song,loaded_model,0)
#         print("predicted index: ",max_index)
        if max_index in mel_track:
            success += 1
        song_count+=1
        print("success rate: ", success/song_count)
    mel_track = []
    print("total success rate: ", success/song_count)

