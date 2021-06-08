import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dense, Concatenate
import os 

def OneHot(file):
  with open(file) as f:
    data = json.load(f)
  result = []
  # print(data.keys())
  for i in data.keys():
    doing = [-1,0,4,8,12,-1]
    passtime = [0,0,0,0,0,0]
    temp = 0
    for j in data[i]:
        doing.append(j[1])
        passtime.append(j[0]-temp)
        temp = j[0]
        # print(temp)
    
    passtime.append(0)
    passtime = passtime[1:]
    dic={'doing':doing,'passtime':passtime}
    # print(dic)
    alldata=pd.DataFrame(dic)
    # print(alldata)
    labelencoder = LabelEncoder()
    data_le=pd.DataFrame(dic)
    data_le['doing'] = labelencoder.fit_transform(data_le['doing'])

    ct = ColumnTransformer([("doing", OneHotEncoder(), [0])], remainder = 'passthrough')
    X = ct.fit_transform(alldata)

    # print(pd.DataFrame(X))
    # print(pd.DataFrame(X)[5:])

    result.append(pd.DataFrame(X[5:]).values)
  return result   


def data_gen():
  files = os.listdir("./train")
  song_data = [] #包很多首歌[[[5(one-hot)],[1(time)],[5(predication one-hot)]]...]
  for f in files:
    print(f)
    # x = OneHot('./train/'+'trt.json')
    # x = OneHot('./train/'+f)
    # t = OneHot('./map_np/'+f)
    # for v in x:
      first_time = True
      one_song = []
      one_song_target = []
      # print (x[0])
      for a in v:
        if first_time:
          first_time = False
          # print("hi")
        else:
          one_song.append(tempx)
          one_song_target.append(a[:-1])
          
          # model.fit([tempx,tempt] , a[:5],epochs=50, batch_size=32)
        tempx = a
      # print (one_song)
      one_song = np.expand_dims(one_song, 0)
      one_song_target = np.expand_dims(one_song_target, 0)
      # one_song = np.array(one_song)
      # one_song_target = np.array(one_song_target)
      
      yield one_song , one_song_target

def vaild_gen():
  files = os.listdir("./vaild")
  song_data = [] #包很多首歌[[[5(one-hot)],[1(time)],[5(predication one-hot)]]...]
  for f in files:
    # print(f)
    x = OneHot('./vaild/'+f)
    # t = OneHot('./map_np/'+f)
    for v in x:
      first_time = True
      one_song = []
      one_song_target = []
      for a in v:
        if first_time:
          first_time = False
          # print("hi")
        else:
          one_song.append(tempx)
          one_song_target.append(a[:-1])
          
          # model.fit([tempx,tempt] , a[:5],epochs=50, batch_size=32)
        tempx = a
      one_song = np.array(one_song)
      one_song_target = np.array(one_song_target)
      # print (one_song.shape)
      yield one_song , one_song_target




model = Sequential()

model.add(LSTM(128,return_sequences=True,  input_shape=(None,6)))
model.add(LSTM(128,return_sequences=True))
model.add(Dense(5,activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])


for item in data_gen():
  (train_data,train_target) = item
  print(train_data.shape)
  history = model.fit(x=train_data,y=train_target,epochs=1)
for item in vaild_gen():
  (vaild_data,vaild_target) = item

data_gen_train = data_gen()
data_gen_vaild = vaild_gen()

# history = model.fit_generator(data_gen_train,shuffle=False,verbose=1)

  