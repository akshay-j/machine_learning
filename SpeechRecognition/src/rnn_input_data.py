#for standard error output
from __future__ import print_function

import os
import sys
import glob

#for reading wav files
import scipy.io.wavfile as wav
import numpy as np

#for calculating MFCC coefficients
from python_speech_features import mfcc

#for required constants
import sound_constants


# this makes text encoding
def convert_text_to_num(ch):
    if ch == "'":
        return sound_constants.SINGLE_QUOTE
    elif ch == ' ':
        return sound_constants.WHITE_SPACE

    return (ord(ch) - 64) % 32


# loading wav file
def process_sound_signal(wavfile):
    sample_rate, sound_signal = wav.read(wavfile)

    # calculating MFCC features
    mfcc_feat = mfcc(sound_signal,sample_rate,numcep=sound_constants.NUM_MFCC_COEFFS)

    # converting into 1-D array of size DATA_SIZE * 1
    mfcc_linear = np.reshape(mfcc_feat,[np.shape(mfcc_feat)[0] * np.shape(mfcc_feat)[1], 1])

    if np.shape(mfcc_linear)[0] > sound_constants.MAX_ROW_SIZE_IN_DATA:
        print ("Number of rows in " + wavfile + " is exceeding maximum permissible row. Increase the value of sound_constants.MAX_ROW_SIZE_IN_DATA param", file = sys.stderr)
        sys.exit(sound_constants.WAV_DATA_TOO_LARGE)

    # pad zeros at the end
    mfcc_linear_with_zeropad = np.zeros([sound_constants.MAX_ROW_SIZE_IN_DATA, sound_constants.MAX_COLUMN_SIZE_IN_DATA])
    mfcc_linear_with_zeropad[:np.shape(mfcc_linear)[0], :np.shape(mfcc_linear)[1]] = mfcc_linear


    return mfcc_linear_with_zeropad


##All functions start from here
def mfcc_and_text_encoding():
    source_dir = os.getcwd()

    if not os.path.exists(sound_constants.ROOT_DIR + sound_constants.DATA_DIR):
        print("Enter correct root directory path !!", file=sys.stderr)
        sys.exit(sound_constants.INVALID_DIR_PATH)

    text_file_encoding = []
    mfcc_data = []


    # get all the wav file names
    for data_dir in sound_constants.ALL_INPUT_DIRS:

        data_directory = sound_constants.ROOT_DIR + sound_constants.DATA_DIR + data_dir

        # check if the working directory exists in the system
        if not os.path.exists(data_directory):
            print(data_directory + " is not a valid directory. Enter correct directory path in the source list !!", file=sys.stderr)
            sys.exit(sound_constants.INVALID_DIR_PATH)

        # change current working directory to input directory
        os.chdir(data_directory)


        wavfile_names = [file for file in os.listdir(data_directory) if file.endswith(sound_constants.AUDIO_FILE_EXTENSION)]
        textfile_names = [file for file in os.listdir(data_directory) if file.endswith(sound_constants.TEXT_FILE_EXTENSION)]
        wav_file_count = len(glob.glob1(data_directory, "*."+sound_constants.AUDIO_FILE_EXTENSION))
        text_file_count = len(glob.glob1(data_directory, "*."+sound_constants.TEXT_FILE_EXTENSION))

        if wav_file_count == 0:
            print ("No " + sound_constants.AUDIO_FILE_EXTENSION + " file present in " + data_directory + ". Exiting program !!", file = sys.stderr)
            sys.exit(sound_constants.NO_WAV_DATA)

        if text_file_count != 1:
            print("No " + sound_constants.TEXT_FILE_EXTENSION + " file present in " + data_directory + ". Exiting program !!", file=sys.stderr)
            sys.exit(sound_constants.NO_TEXT_DATA)


        # extract wav files
        for count in range(wav_file_count):
            mfcc_data.append(process_sound_signal(wavfile_names[count]).astype(dtype = np.float32, copy = False))


        # encode text
        text_file_data = []
        new_list = []
        with open(textfile_names[0], "r") as fileobj:
            text_file_data = fileobj.read().splitlines()

            for strl in text_file_data:
                new_list.append(strl[14:])

        for stri in new_list:
            temp_str = map(convert_text_to_num, stri)
            temp_str = np.array(temp_str, dtype = np.float32)
            temp_str_with_zeropad = np.zeros([sound_constants.MAX_ROW_SIZE_IN_TXT])
            temp_str_with_zeropad[:np.shape(temp_str)[0]] = temp_str

            arr = np.reshape(temp_str_with_zeropad, (-1, 1))
            text_file_encoding.append(arr)

        print (data_directory + " processed !!")

    # switch to program directory
    os.chdir(source_dir)

    return mfcc_data, text_file_encoding
