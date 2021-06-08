from audioprocess import *
from mapprocess import *
import json

root_path="D:/osu/Songs"
allFolderList = os.listdir(root_path) #讀取所有的資料夾(歌曲)

for folder in allFolderList: #讀取資料夾的檔案
    path = root_path+"/"+folder

    #儲存圖譜(dictionary)
    difficulty_list, osu_maps = getMaps(path)
    if len(difficulty_list)==0:
        continue
    map_dict= dict()
    for i, d in enumerate(difficulty_list):
        map_dict[d]=osu_maps[i]
    save_file = open("preprocessing/map_np/"+folder+".json", "w")
    json.dump(map_dict, save_file)
    save_file.close()

    #儲存音檔(nparray)
    audio_path = getAudioName(path) #取得音檔
    wav_path = audio2wav(audio_path,"preprocessing/") #轉成wav(前置處理主要處理wav檔)
    filter_bank = getMelFB(wav_path)
    cnn_data = getCNNformat(filter_bank)
    cnn_data = getNormalization(cnn_data)
    np.save('preprocessing/audio_np/'+folder,cnn_data)