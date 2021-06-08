import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import os 

def plot_compare(history, steps=-1):
    if steps < 0:
        steps = len(history.history['acc'])
    acc = smooth_curve(history.history['acc'][:steps])
    val_acc = smooth_curve(history.history['val_acc'][:steps])
    loss = smooth_curve(history.history['loss'][:steps])
    val_loss = smooth_curve(history.history['val_loss'][:steps])

    plt.figure(figsize=(6, 4))
    plt.plot(loss, c='#0c7cba', label='Train Loss')
    plt.plot(val_loss, c='#0f9d58', label='Val Loss')
    plt.xticks(range(0, len(loss), 5))
    plt.xlim(0, len(loss))
    plt.title('Train Loss: %.3f, Val Loss: %.3f' % (loss[-1], val_loss[-1]), fontsize=12)
    plt.legend()

    plt.figure(figsize=(6, 4))
    plt.plot(acc, c='#0c7cba', label='Train Acc')
    plt.plot(val_acc, c='#0f9d58', label='Val Acc')
    plt.xticks(range(0, len(acc), 5))
    plt.xlim(0, len(acc))
    plt.title('Train Accuracy: %.3f, Val Accuracy: %.3f' % (acc[-1], val_acc[-1]), fontsize=12)
    plt.legend()

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
  files = os.listdir("StepSelection/train")
  song_data = [] #包很多首歌[[[5(one-hot)],[1(time)],[5(predication one-hot)]]...]
  for f in files:
    # print(f)
    x = OneHot('StepSelection/train/'+f)
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
      one_song = np.expand_dims(one_song, 0)
      one_song_target = np.expand_dims(one_song_target, 0)
      # one_song = np.array(one_song)
      # one_song_target = np.array(one_song_target)
      # print (one_song.shape)
      yield one_song , one_song_target

def vaild_gen():
  files = os.listdir("StepSelection/vaild")
  song_data = [] #包很多首歌[[[5(one-hot)],[1(time)],[5(predication one-hot)]]...]
  for f in files:
    # print(f)
    x = OneHot('StepSelection/vaild/'+f)
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
      one_song = np.expand_dims(one_song, 0)
      one_song_target = np.expand_dims(one_song_target, 0)
      # one_song = np.array(one_song)
      # one_song_target = np.array(one_song_target)
      # print (one_song.shape)
      yield one_song , one_song_target




##訓練用模型
inputs = Input(shape=(None,6))
lstm1 = LSTM(128,return_sequences=True, return_state=True)
lstm1_outputs, _, _ = lstm1(inputs,initial_state=None)
lstm2 = LSTM(128, return_sequences=True, return_state=True)
lstm2_outputs, _, _ = lstm2(lstm1_outputs, initial_state=None)
dense = Dense(5,activation='softmax')
outputs = dense(lstm2_outputs)
model = Model(inputs,outputs)
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])


for item in data_gen():
  (train_data,train_target) = item
  # print(train_data.shape)

for item in vaild_gen():
  (vaild_data,vaild_target) = item

data_gen_train = data_gen()
#data_gen_vaild = vaild_gen()

# history = model.fit_generator(data_gen_train,shuffle=False,verbose=1)
history = model.fit(train_data,train_target,epochs=250,validation_data=(vaild_data,vaild_target))

#plot_compare(history)

##預測用模型
# 定義LSTM1 的 state input
lstm1_state_input_h = Input(shape=(128,))
lstm1_state_input_c = Input(shape=(128,))
lstm1_states_inputs = [lstm1_state_input_h, lstm1_state_input_c]

# 定義解碼器 LSTM1 模型
lstm1_outputs, lstm1_state_h, lstm1_state_c = lstm1(
    inputs, initial_state=lstm1_states_inputs)

# 定義LSTM2 的 state input
lstm2_state_input_h = Input(shape=(128,))
lstm2_state_input_c = Input(shape=(128,))
lstm2_states_inputs = [lstm2_state_input_h, lstm2_state_input_c]

# 定義解碼器 LSTM2 模型
lstm2_outputs, lstm2_state_h, lstm2_state_c = lstm2(
    lstm1_outputs, initial_state=lstm1_states_inputs)


# 以編碼器的記憶狀態 h 及 c 為解碼器的記憶狀態  
lstm1_states = [lstm1_state_h, lstm1_state_c]
lstm2_states = [lstm2_state_h, lstm2_state_c]
outputs = dense(lstm2_outputs)
selection_model = Model(
    [inputs]+lstm1_states_inputs+lstm2_states_inputs,
    [outputs]+lstm1_states+lstm2_states
)


selection_model.save("./StepSelection.h5")
#loss: 0.2831 - acc: 0.6041