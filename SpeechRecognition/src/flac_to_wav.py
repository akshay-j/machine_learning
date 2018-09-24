import os
from pydub import AudioSegment

# constants
ROOT_DIR = "/home/arpita/Documents/akshay/"
SRC_DIR = ROOT_DIR + "libri-speech-100hrs/LibriSpeech/train-clean-100/"
DST_DIR = ROOT_DIR + "all_wav_files/"
DIR_INFO_FILE = "all_dirs_list.txt"

# read text file data about directory
file_obj = open(ROOT_DIR + DIR_INFO_FILE, "r")
dir_list = []
dir_list.append(file_obj.read())

new_list = []
for dir in dir_list:
    new_list = dir.split('\n')

new_list = new_list[:-1]

for dir in new_list:
    FINAL_DST_DIR = DST_DIR + dir
    dir = SRC_DIR + dir
    if os.path.isdir(dir):
        flac_names = [file for file in os.listdir(dir) if
                         file.endswith("flac")]

        if not os.path.exists(FINAL_DST_DIR):
            print "Creating directory:" + FINAL_DST_DIR
            os.makedirs(FINAL_DST_DIR)

        #switch directory to where you want to save files
        os.chdir(FINAL_DST_DIR)


        flac_names_new = []

        # conversion from flac to wav
        for flac_file in flac_names:
            flac_names_new.append(flac_file.split('.')[0])

        for audio in flac_names_new:
            flac_audio = AudioSegment.from_file(dir + audio + ".flac", "flac")
            print "writing " + audio + ".wav to " + os.getcwd()
            flac_audio.export(audio + ".wav", format="wav")
