import numpy as np
import zipfile

# 注意事項
# 執行後便將指定歌名的.osu與audio.wav檔案壓縮成osz格式
# 生成的.osu檔與壓縮成功後的.osz檔將會儲存於osugenerator/osuMaps/裡面
# 若歌名為非英文，壓縮至osu會產生亂碼，但不影響執行

def osuFormatDictGenerator(artist = "default", title = "default", difficulty = "Normal", source = "default", tags = "default", HP = 5, CS = 5, OD = 5, AR = 5):
    result = dict()

    # General
    temp = dict()
    temp["AudioFilename"] = "audio.wav"
    temp["AudioLeadIn"] = 0
    temp["PreviewTime"] = -1
    temp["Countdown"] = 0
    temp["SampleSet"] = "Normal"
    temp["StackLeniency"] = 0.7
    temp["Mode"] = 1
    temp["LetterboxInBreaks"] = 0
    temp["WidescreenStoryboard"] = 0
    result["General"] = temp
    
    # Editor
    temp = dict()
    temp["DistanceSpacing"] = 1
    temp["BeatDivisor"] = 4
    temp["GridSize"] = 32
    temp["TimelineZoom"] = 0.5
    result["Editor"] = temp

    # Metadata
    temp = dict()
    temp["Title"] = title
    temp["TitleUnicode"] = title
    temp["Artist"] = artist
    temp["ArtistUnicode"] = artist
    temp["Creator"] = "taiko-sun"
    temp["Version"] = difficulty
    temp["Source"] = source
    temp["Tags"] = tags
    temp["BeatmapID"] = 0
    temp["BeatmapSetID"] = -1
    result["Metadata"] = temp

    # Difficulty
    temp = dict()
    temp["HPDrainRate"] = HP
    temp["CircleSize"] = CS
    temp["OverallDifficulty"] = OD
    temp["ApproachRate"] = AR
    temp["SliderMultiplier"] = 1
    temp["SliderTickRate"] = 1
    result["Difficulty"] = temp

    # Events
    temp = dict()
    result["Events"] = temp
    # TimingPoints
    result["TimingPoints"] = "0,-100,4,1,0,100,1,0"
    # HitObjects
    result["HitObjects"] = ""

    return result

def osuGenerator(rhythmList, fileName = "default", audioFileName = "audio.wav", oszFileName = "generatedBeatmap"):
    osuFormat = osuFormatDictGenerator(title = fileName)

    with open("osuGenerator//osuMaps//"+fileName+".osu", "w", encoding="utf-8") as f:
        f.write("osu file format v14 \n\n")
        for key, value in osuFormat.items():
            f.write("[" + key + "]\n")
            if type(value) == type(dict()):
                for det_key, det_value in value.items():
                    f.write(det_key + ": " + str(det_value) + "\n")
            elif key == "TimingPoints":
                f.write(value+"\n")
            elif key == "HitObjects":
                for miliSec, beatType in rhythmList:
                    f.write("255,194,"+str(miliSec)+",1,"+str(beatType)+",0:0:0:0:\n")
            f.write("\n")
    
    with zipfile.ZipFile("osuGenerator//osuMaps//"+oszFileName+".osz", 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write("osuGenerator//osuMaps//"+fileName+".osu",\
             arcname=osuFormat["Metadata"]["Artist"] + " - " + osuFormat["Metadata"]["Title"] + " (" + osuFormat["Metadata"]["Creator"] + ") [" +\
                 osuFormat["Metadata"]["Version"] +  "].osu")
        zf.write("generate_map//" + fileName + "//"+AUDIO_FILE_NAME, arcname="audio.wav")

SONG_NAME = "歌ってみたグッバイ宣言 Kotone(天神子兎音)"
AUDIO_FILE_NAME = "audio.wav"
DIFFICULTY_NUMBER = "3"

with open("generate_map\\"+SONG_NAME+"\\"+DIFFICULTY_NUMBER, 'r') as f:
    temp = f.readlines()
    rst = list()
    for ele in temp:
        tempList = list()
        tempList.append(ele.split(" ")[0])
        tempList.append(ele.split(" ")[1].split("\n")[0])
        rst.append(tempList)

    print(rst)

    osuGenerator(rst, fileName=SONG_NAME, audioFileName=AUDIO_FILE_NAME)
