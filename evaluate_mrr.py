
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
from time import time
import os
import pypianoroll as pr
import numpy as np
import random
from random import shuffle
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

tf.keras.backend.set_session(sess)


# In[3]:


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


# In[4]:


#load midi files with melody tracks
mel_roll = []
def load_mel_midi():
     for root, dirs, files in os.walk("./midi_with_mel/", topdown=False):
        for name in files:
            full_name = os.path.join(root,name)
            if full_name.endswith('mid'):
                temp = pr.parse(full_name)
                temp.name = name.split('.')[0]
                mel_roll.append(temp)
# load_mel_midi()


# In[5]:


#handle the prediction returned from the model, return maximum likelihood track num
def handle_prediction_full_song(prediction,random,length):
    label = []
    if random != 0:
        prediction = prediction.reshape(int(prediction.shape[0]/random),random,1)
    elif random == 0:
        prediction = prediction.reshape(int(prediction.shape[0]/length),length,1)
    avg_score = np.average(prediction,axis=1)
    print("avg_score shape ", avg_score.shape())
    max_index = np.argmax(avg_score,axis = 0)
#     for x in range(len(song.tracks)):
#         if avg_score[x]:
#             song.tracks[x].name = "Melody"
    
    return max_index


# In[12]:


# handle the prediction and return [(track score, track index)]
def handle_prediction(prediction,random,length):
    label = []
    if random != 0:
        prediction = prediction.reshape(int(prediction.shape[0]/random),random,1)
    elif random == 0:
        prediction = prediction.reshape(int(prediction.shape[0]/length),length,1)
    avg_score = np.average(prediction,axis=1).T[0]
    pos = [x for x in range(len(avg_score))]
    scores = sorted(zip(avg_score, pos), reverse=True)
    print(scores)
    return scores


# In[28]:


# predict single song and return 
def predict_song(song,loaded_model,random):
    img_rows = 128
    img_cols = 500
    store = []
    temp = []
    temp_song = song
    #iterate over tracks and randomly sample 3 subarrays in a track if random is set
    #if random is not set, iterate over the track with timestep = 500, and store them in store
    for track in temp_song.tracks:
        if random != 0:
            random_ints = random.sample(range(0,track.pianoroll.shape[0]-500),random)
            for x in range(0,random):
                temp_array = np.array(track.pianoroll[random_ints[x]:random_ints[x]+500,:])
                temp.append((temp_array).swapaxes(0,1))
        # else if not random
        elif random == 0:
            # iterate over track, with timestep = 500
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
    scores = handle_prediction(prediction,random,length)
    return scores


# In[44]:


def main_function():
    song_count = 0
    success = 0
    mel_track = []
    mrr_score = 0
    all_scores = []
    melody_track_nums = []
    predicted_tracks = []
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for root,dirs,files in os.walk("./lpd_cleansed/", topdown=False):
        for name in files:
            full_name = os.path.join(root,name)
            if full_name.endswith('.npz'):
                print(full_name)
                if song_count == 2:
                    return all_scores, true_pos, false_pos, true_neg, false_neg
                try:
                    song = pr.load(full_name)
                    tracks = song.tracks
                    shuffle(tracks)
                    track_num = len(tracks)
                    for x in range(len(tracks)):
                        if tracks[x].name.strip().lower() in ['melody','vocal','mel']:
                            mel_track.append(x)
                            print("melody track: ",x, tracks[x].name)
                    if len(mel_track) == 0:
                        print("no melody track found")
                        continue
                    else:
                        melody_track_nums.append(len(mel_track))
                    #predict the song and return (track_score, track_index)
                    scores = predict_song(song,loaded_model,0)
                    #save the scores to "all_scores" and store the tracks that have a score above 0.5 in predicted_tracks
                    for tuples in scores:
                        if tuples[0] >= 0.5:
                            predicted_tracks.append(tuples[1])
                        all_scores.append(tuples[0])
                    #compute the false pos/neg
                    for x in predicted_tracks:
                        if x in mel_track:
                            true_pos += 1
                            track_num -= 1
                        else:
                            false_pos += 1
                            track_num -= 1
                    for x in mel_track:
                        if x not in predicted_tracks:
                            false_neg += 1
                            track_num -= 1
                    true_neg += track_num
                    #dealing with mrr_score
#                     for x in range(len(max_index)):
#                         if max_index[x][1] in mel_track:
#                             print("found matched track at rank ", x, " at track", max_index[x][1])
#                             mrr_score += 1/(x+1)
#                             break
                    #success if predicted melody track is in melody track
#                     if max_index in mel_track:
#                         success += 1
                    song_count+=1
                    print("processed songs: ", song_count)
                except Exception as error:
                    print("Stucked file: "+ full_name)
                    raise(error)
                    pass
                scores = 0
                mel_track = []
                print(true_pos, false_pos, true_neg, false_neg)
    return (all_scores, true_pos, false_pos, true_neg, false_neg)
    


# In[45]:


all_scores,tp,fp,tn,fn = main_function()


# In[47]:


print(all_scores)
print(tp, fp, tn, fn)
total_tracks = tp+fp+tn+fn
print("tp = ", tp/total_tracks)
print("fp = ", fp/total_tracks)
print("tn = ", tn/total_tracks)
print("fn = ", fn/total_tracks)


# In[53]:


def count_elements(seq,parts):
    hist = {}
    get = hist.get
    for i in seq:
        i_floor = math.floor(i/parts)
        hist[i_floor] = get(i_floor,0)+1
    return hist


# In[56]:


parts = 0.00001
counted = count_elements(all_scores,parts)
fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(111)
ax.bar(counted.keys(), counted.values(),width = 10, color='g')


plt.xlim(0,1/parts)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
fig.savefig(fname = "plot"+str(parts)+".png")


# In[52]:


with open("output.txt","w") as f:
    f.write("total tracks: "+ str(total_tracks)+'\n')
    f.write("true positive: "+ str(tp/total_tracks)+'\n')
    f.write("true negative: "+ str(tn/total_tracks)+'\n')
    f.write("false positive: "+ str(tn/total_tracks)+'\n')
    f.write("false negative: "+ str(tn/total_tracks)+'\n')
    f.close()

