from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import librosa
import scipy
import os

def getAudioName(path):
    allFileList = os.listdir(path)
    for file in allFileList: #找到.osu檔案
        if file.endswith(".osu"):
            osu_file=file
            break
    
    audio_name=""
    with open(path+"/"+osu_file, "r",encoding="utf-8") as f:
        for line in f:
            if "AudioFilename" in line: #找到歌名
                audio_name=line.split(" ",1)[1][:-1]
    return path+"/"+audio_name

def audio2wav(audio_path,save_path=""): #會回傳路徑
    #判斷檔案類型
    audio_type="wav" #預設為wav
    if audio_path.endswith("mp3"):
        audio_type="mp3"
    elif audio_path.endswith("ogg"):
        audio_type="ogg"
    elif audio_path.endswith("flv"):
        audio_type="flv"
    elif audio_path.endswith("mp4"):
        audio_type="mp4"
    elif audio_path.endswith("wma"):
        audio_type="wma"
    elif audio_path.endswith("aac"):
        audio_type="acc"

    #取得音檔名稱
    # index=audio_path.rfind("/")
    # if index != -1:
    #     audio_name=audio_path[index+1:-4]
    # else:
    #     audio_name=audio_path[:-4]

    #儲存至路徑
    sndObj=AudioSegment.from_file(audio_path,audio_type)
    sndObj.export(save_path+"/audio.wav", format="wav")
    return save_path+"/audio.wav"

def getMelFB(path,window_lengths=[23,46,93],stride_length=10,n_mels=80):
    rate, data = read(path)
    if len(data.shape)==2:
        data = data.T
        data = (data[0]+data[1])/2
    elif len(data.shape)==1:
        data = data/2
    ms_len = rate/1000 #一毫秒的取樣數

    logmelspecs=[]
    for window_length in window_lengths:
        n_fft = int(window_length*ms_len)
        hop_length = int(stride_length*ms_len)
        stft = np.abs(librosa.stft(data,n_fft=n_fft,hop_length=hop_length,window=scipy.signal.hamming))**2
        melspec = librosa.feature.melspectrogram(S=stft,n_mels=n_mels) #取得mel的filterbank
        #melspec = librosa.feature.melspectrogram(data,sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels) #取得mel的filterbank
        logmelspec = librosa.amplitude_to_db(melspec) #取對數
        logmelspec = logmelspec.T
        logmelspecs.append(logmelspec)
    return logmelspecs

def getCNNformat(data,height=7): #資料 堆疊高度
    cnnGates=[] #儲存堆疊過後的三個通道
    for d in data:
        cnnData=[] #堆疊前後各n個frame
        data_len = len(d)
        #頭尾n個都不算
        for i in range(0,data_len):
            if i < height:
                continue
            if i+height < data_len:
                cnnData.append(d[i-height:i+height+1])
            else: #超過長度
                break
        cnnData = np.array(cnnData)
        cnnGates.append(cnnData)
    
    #三個通道排列處理
    returnData=np.zeros(shape=(cnnGates[0].shape[0],cnnGates[0].shape[1],cnnGates[0].shape[2],3))
    for i in range(len(returnData)):
        returnData[i]=np.dstack([cnnGates[0][i],cnnGates[1][i],cnnGates[2][i]])
    return returnData

def getNormalization(data, target_max=1, traget_min=-1):
    data = np.float32(data)
    mean = np.mean(data) # 0為平均 1為標準差
    data = data-mean
    std = np.std(data)
    data = data/std
    # MAX = np.max(data)
    # MIN = np.min(data)
    # data = (data-MIN)/(MAX-MIN)*(target_max-traget_min)+traget_min
    return data

if __name__ == '__main__':
    ### 轉wave
    #audio2wav("Agehachou.mp3","preprocessing")
    audio2wav("Agehachou.mp3","preprocessing/")

    filter_bank = getMelFB("preprocessing/audio.wav")
    print(filter_bank[0].shape,filter_bank[1].shape,filter_bank[2].shape)

    cnn_data = getCNNformat(filter_bank)
    print(cnn_data.shape)

    cnn_data = getNormalization(cnn_data)
    print(np.max(cnn_data),np.min(cnn_data))

    # print(cnn_data.shape)
    # name = getAudioName("D:/osu/Songs/1349645 Nekomata Master vs HuMeR - BUZRA")
    # print(name)