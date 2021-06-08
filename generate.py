from preprocessing.audioprocess import *
import librosa
from librosa.util import peak_pick
from tensorflow import keras
from scipy import signal
import numpy as np
import os

class MapGenerator:
    def __init__(self):
        self.music_path=""
        self.save_path=""
        self.placement_model_path="model/StepPlacement97(best_loss).h5"
        self.selection_model_path="model/StepSelection1000.h5"
        self.placement_model = keras.models.load_model(self.placement_model_path)
        self.selection_model = keras.models.load_model(self.selection_model_path)

    def generate(self,music_path,save_path,level=0):
        self.music_path=music_path
        self.save_path=save_path
        feature = self.__getFeature()
        peaks = self.__getPeaks(feature,level)
        y = self.__getGenMap(peaks)
        self.__saveMap(y,level)

    def generateAllDiff(self,music_path,save_path):
        self.music_path=music_path
        self.save_path=save_path
        feature = self.__getFeature()
        for i in range(5):
            level = i
            peaks = self.__getPeaks(feature,level)
            y = self.__getGenMap(peaks)
            self.__saveMap(y,level)
        
    def __saveMap(self,m,level):
        f = open(self.save_path+"/"+str(level),'w')
        for obj in m:
            f.writelines(str(obj[0])+"0 "+str(obj[1])+"\n")
        f.close

    def __getFeature(self):
        music_name = self.music_path.split("/")[-1][:-4]
        self.save_path = self.save_path+music_name
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        audio2wav(self.music_path,self.save_path)
        filter_bank = getMelFB(self.save_path+"/"+"audio.wav")
        cnn_data = getCNNformat(filter_bank)
        x = getNormalization(cnn_data)
        return x

    def __getPeaks(self,x,level):
        y = np.zeros(shape=(x.shape[0],5))
        y[:,level]=1

        result = self.placement_model.predict([x,y])

        data = []
        for r in result:
            data.append(r[0])
        data=np.array(data)
        data = data*1000
        data = data.astype(int)
        win = signal.windows.hamming(50)
        x = signal.convolve(data,win,mode='same')/sum(win)
        peaks, _ = signal.find_peaks(x, prominence=6-level)
        return peaks

    def __getGenMap(self,peaks):
        x = np.zeros((1, 1, 6))
        x[0][0]=np.array([1,0,0,0,0,0])
        state=np.zeros(shape=(1,128))
        lstm1_state=[state,state]
        lstm2_state=[state,state]
        result=[]
        for i, peak in enumerate(peaks):
            if i==0:
                x[0][0][-1]=peak+8
            else:
                x[0][0][-1]=peak-peaks[i-1]
            y,lstm1_state[0], lstm1_state[1], lstm2_state[0], lstm2_state[1]=\
                        self.selection_model.predict([x]+[lstm1_state]+[lstm2_state])
            x[0][0]=np.array([0,0,0,0,0,0])
            x[0][0][np.argmax(y)]=1
            result.append(np.array([peak+8,(np.argmax(y)-1)*4]))
        return result

if __name__=="__main__":
    # x = getFeature("Agehachou.mp3","generate_map/")
    # x = getPeaks(x,0)
    # m = getGenMap(x)
    mg = MapGenerator()
    mg.generateAllDiff("D:/myMusic/youtube轉出音樂/歌ってみたベノム かいりきベアKotone(天神子兎音)cover.mp3","generate_map/")