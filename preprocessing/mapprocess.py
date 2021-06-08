from osu_sr_calculator import calculateStarRating
import heapq
import os

def getMapObj(path): #傳入osu檔案路徑
    with open(path,"r",encoding="utf-8") as f:
        isFound=False #是否找到HitObject開頭
        hit_objects=[]
        for line in f:
            if isFound:
                object_detail = line.split(",") #原本物件的細節
                if len(object_detail) != 6: #輪盤物件
                    # for i in range(int(object_detail[2]),int(object_detail[5])+1,10):
                    #     hit_objects.append([i//10,20])
                    continue
                mill_sec = int(object_detail[2])//10 #第幾毫秒(以10毫秒取)
                obj = int(object_detail[4]) #何種物件
                hit_objects.append([mill_sec, obj])

            if ("[HitObjects]" in line) and not isFound: #紀錄HitObject從第幾行開始
                isFound=True
    return hit_objects

def getMaps(path):
    allFileList = os.listdir(path)
    osu_files = [file for file in allFileList if file.endswith(".osu")] #找.osu檔

    difficulty_list=[]
    osu_maps=[] #儲存圖譜
    for name in osu_files:
        filepath=path+"/"+name

        starRating = VerifyDifficulty(name)

        if starRating in difficulty_list or starRating == -1: #重複與無法辨識的難度捨去
            continue
        difficulty_list.append(starRating)
        osu_maps.append(getMapObj(filepath))
    return difficulty_list, osu_maps

def VerifyDifficulty(name):
    name = name.lower()
    if "easy" in name or "kantan" in name:
        return 0
    elif "normal" in name or "futsuu" in name:
        return 1
    elif "hard" in name or "muzukashii" in name:
        return 2
    elif "insane" in name or ("oni" in name and "inner" not in name):
        return 3
    elif "expert" in name or "inner oni" in name:
        return 4
    else:
        return -1

if __name__=='__main__':
    #hit_objects = getMapObj("D:/osu/Songs/1079752 Eve - Last Dance/Eve - Last Dance (Skull Kid) [HiroK's Oni].osu")
    #print(hit_objects)
    osu_maps = getMaps("D:/osu/Songs/1079752 Eve - Last Dance")
    print(osu_maps)
