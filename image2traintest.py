import numpy as np
import argparse
import csv
import shutil
from os import path,mkdir
import os

MAX_COORDINATES = 1024

def main(current_file_dir, flags):
    #Remove previous train and validation directory if it exists
    if path.exists(flags.train_dir):
        shutil.rmtree(flags.train_dir)
    if path.exists(flags.validation_dir):
        shutil.rmtree(flags.validation_dir)

    #Make new training and validation directory with permission Read/write/execute permissions for owner and R,W permissions for others
    mkdir(flags.train_dir, 0o755 )
    mkdir(flags.validation_dir, 0o755 )

    print("Processing images from {} folder to training folder".format(flags.mode))
    # Copy the corresponding files labelled in train and validation text from images source accordingly
    with open(flags.input_train, 'r') as ft:
        #reader = csv.reader(ft)
        #next(reader, None) #Skip the headers
        cnt = 0
        reader = ft.readlines()
        for line in reader:
            temp_file = line.strip().split(" ")[0]
            #print(temp_file)
            source_file = os.path.join(current_file_dir, flags.mode, str(temp_file))
            dest_file = os.path.join(current_file_dir, flags.train_dir, str(temp_file))
            #source_file = ''.join([current_file_dir,str(temp_file)])
            #dest_file = ''.join([flags.train_dir,str(temp_file)])
            if path.exists(source_file):
                dest = shutil.copyfile(source_file, dest_file)
                cnt+=1
            check_file = temp_file
        print("Copied {} images into training folder".format(cnt))
    #write in file /anntations/val.txt
    print("Processing images to validation folder")

    with open(flags.input_validation, 'r') as fv:
        #reader = csv.reader(fv)
        #next(reader, None) #Skip the headers
        cnt = 0
        check_file = "uninitialised"
        reader = fv.readlines() #Reads whole file and split by line
        for line in reader:
            #print(line)
            temp_file = line.strip().split(" ")[0]
            print(temp_file)
            source_file = os.path.join(current_file_dir, flags.mode, str(temp_file))
            dest_file = os.path.join(current_file_dir, flags.validation_dir, str(temp_file))
            #source_file = ''.join([source,str(temp_file)])
            #dest_file = ''.join([flags.validation_dir,str(temp_file)])
            if path.exists(source_file):
                dest = shutil.copyfile(source_file, dest_file)
                #print(source_file)
                cnt+=1
            check_file = temp_file
        print("Copied {} images into validation folder".format(cnt))

if __name__ == '__main__':
    """
    THis program copies the images to training and validation folder based on the list of training and validation annotation file generated
    """
    #Get current directory of this file, which is Vedai
    current_file_dir = os.path.dirname(os.path.realpath("__file__"))

    DEFAULT_INPUT_TRAIN_FILE = os.path.join(current_file_dir,"annotations","train_vedai.csv")
    DEFAULT_INPUT_VAL_FILE = os.path.join(current_file_dir,"annotations","val_vedai.csv")
    DEFAULT_TRAIN_DIR = os.path.join(current_file_dir,"train")
    DEFAULT_VAL_DIR = os.path.join(current_file_dir,"validation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='CO') #visible (co), ir or visible+ir (coir)
    parser.add_argument("--input_train", default = DEFAULT_INPUT_TRAIN_FILE) #path to original annotation file
    parser.add_argument("--input_validation", default = DEFAULT_INPUT_VAL_FILE) #path to original annotation file
    parser.add_argument("--train_dir", default = DEFAULT_TRAIN_DIR) #path to original annotation file
    parser.add_argument("--validation_dir", default = DEFAULT_VAL_DIR) #path to original annotation file
    flags = parser.parse_args()
    main(current_file_dir,flags)