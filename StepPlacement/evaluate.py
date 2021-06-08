import numpy as np
import json
import os

model_path = "model/StepPlacement97(best_loss).h5"

root_path="D:/DeepLearning-workplace/taiko_generate/preprocessing/audio_np"
audio_path = os.listdir(root_path)
audio_path = [ root_path+"/"+ad for ad in audio_path]

root_path="D:/DeepLearning-workplace/taiko_generate/preprocessing/map_np"
map_path = os.listdir(root_path)
map_path = [ root_path+"/"+mp for mp in map_path]

t_audio_path = audio_path[:int(len(audio_path)*0.9)]
t_map_path = map_path[:int(len(map_path)*0.9)]

v_audio_path = audio_path[int(len(audio_path)*0.9):]
v_map_path = map_path[int(len(map_path)*0.9):]

print(len(t_audio_path),len(t_map_path),len(v_audio_path),len(v_map_path))

from tensorflow import keras
import librosa
from librosa.util import peak_pick
import matplotlib.pyplot as plt
from scipy import signal

model = keras.models.load_model(model_path)

F_measure_list=[]
for i,ap in enumerate(v_audio_path):
    audio=np.load(ap)
    f = open(v_map_path[i], "r") #讀所有的map
    maps = json.load(f)
    f.close()
    
    for key in maps.keys():
        
        dif = np.zeros(shape=(audio.shape[0],5))
        dif[:,int(key)]=1
        result = model.predict([audio, dif])

        data = []
        for r in result:
            data.append(r[0])
        data=np.array(data)

        data = data*1000
        data = data.astype(int)

        win = signal.windows.hamming(50)
        x = signal.convolve(data,win,mode='same')/sum(win)
        
        distance = 5//(int(key)+1)
        peaks, _ = signal.find_peaks(x, distance=distance, prominence=6-int(key)) #6-int(key)
        
        
        
        # 取出實際map之節拍毫秒值
        realMiliSecs = list()
        for ele in maps[key]:
            realMiliSecs.append(ele[0])

        # 誤差區間值
        DET_RANGE = 10
        # 計算TP, FP, FN score以得出精準度
        tpScore, fpScore, fnScore, previousRealMiliSecs, hitFlag = 0, 0, 0, 0, False
        for guessMiliSecs in peaks:
            # 取得與猜測時間點最近的實際時間點
            closestRealMiliSecs = min(realMiliSecs, key = lambda x:abs(x-guessMiliSecs))
            if previousRealMiliSecs != closestRealMiliSecs:
                if not hitFlag:
                    fnScore += 1
                hitFlag = False
            #print([closestRealMiliSecs, guessMiliSecs])
            if (closestRealMiliSecs - 8) + DET_RANGE > guessMiliSecs > (closestRealMiliSecs - 8) - DET_RANGE:
                #print('hit')
                tpScore += 1
                hitFlag = True
            else:
                fpScore += 1
            previousRealMiliSecs = closestRealMiliSecs

        F_measure = 2 * tpScore / (2 * tpScore + fpScore + fnScore)
        F_measure_list.append(F_measure)
        print(F_measure)

print(sum(F_measure_list)/len(F_measure_list))