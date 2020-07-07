import numpy as np
import argparse
import os

MAX_COORDINATES = 1024

def getImageFileList(input_file,training_files):
    """
    Processes the annotation file and returns a set containing filenames of images.Assumes the annotation file lists the file names in order
    """
    count_train = 1
    s=set()
    with open(input_file, 'r') as input_file:
        line = input_file.readline()#Read a single line
        while line and len(s)<=training_files+1:
            #The use of split_factor + 1
            information = line.strip().split(" ")
            #Extract the filename
            reference_file=str(information[0])
            s.add(reference_file)
            if len(s)==training_files+1:
                #Remove the last element added at the first instance
                s.remove(reference_file)
                print("Processed {} entries with {}/{} total unique files for training".format(count_train,len(s),number_files))
                break
            count_train+=1
            line = input_file.readline() #Read subsequent lines
    return s

#reshape box
def getBoxCoordinates(array):
    """
    Process the coordinates of the boxes to be within valid range
    """
    Xs = [float(e) for e in array[4:8]]
    Ys = [float(e) for e in array[8:12]]
    tmp_Xmin = min(max(min(Xs), 0), MAX_COORDINATES)
    tmp_Xmax = min(max(max(Xs), 0), MAX_COORDINATES)
    tmp_Ymin = min(max(min(Ys), 0), MAX_COORDINATES)
    tmp_Ymax = min(max(max(Ys), 0), MAX_COORDINATES)
    return str(tmp_Xmin), str(tmp_Xmax), str(tmp_Ymin), str(tmp_Ymax)

def main(flags,training_file_list):
    """
    Case of typical train validation splits
    Reads and extracts the original annotation file, extract and process to 2 separate annotated files:training and validation
    """
    with open(flags.input_file, 'r') as fr:
        #Creates new training dataset and writes the content
        line = fr.readline() #Read the first line
        ptr = '00000000'
        row_count = 0
        row_val = 0
        print("Writing to {}".format(flags.train_file))
        with open(flags.train_file, 'w+') as fw:
            count_unique_files = 0 #Counter for unique files
            #ptr = '00000000'
            row_train = 0
            #Define a base template for each row of records to be return starting with imagefilename
            chars = [ptr + '_' + flags.mode.lower() + '.png']
            #line = fr.readline() #Read the first line
            while line and (ptr in training_file_list):
                new_chars = line.strip().split(" ")
                if len(new_chars) != 15:
                    print("Checking contents")
                    print("Wrong number of elements in line :", cnt)
                    print("Further Information: {}".format(new_chars))
                if new_chars[0] == ptr:
                    Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                    new_chars = [new_chars[12], Xmin, Ymin, Xmax, Ymax]
                    #Extend the list of information of bounding boxes for each image (class, xmin,ymin,xmax,ymax)
                    chars.extend(new_chars)
                else:
                    new_line = ' '.join(chars) + "\n" #Write information as new line
                    fw.write(new_line)
                    #Assign new ptr for next file
                    ptr = new_chars[0]
                    idx = ptr + '_' + flags.mode.lower() + '.png'
                    Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                    chars = [idx, new_chars[12], Xmin, Ymin, Xmax, Ymax]
                    count_unique_files += 1
                row_train += 1
                row_count +=1
                print("Training file count : {}".format(row_train))
                #Read the next line
                line = fr.readline()
        print("Writing to {}".format(flags.val_file))
        with open(flags.val_file, 'w+') as fw:
            while line:
                new_chars = line.strip().split(" ")
                if len(new_chars) != 15: 
                    print("Wrong number of elements in line :", cnt)
                if new_chars[0] == ptr:
                    Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                    new_chars = [new_chars[12], Xmin, Ymin, Xmax, Ymax]
                    chars.extend(new_chars)
                else:
                    new_line = ' '.join(chars) + "\n"
                    fw.write(new_line)
                    ptr = new_chars[0]
                    idx = ptr + '_' + flags.mode.lower() + '.png'
                    Xmin, Xmax, Ymin, Ymax = getBoxCoordinates(new_chars)
                    chars = [idx, new_chars[12], Xmin, Ymin, Xmax, Ymax]
                row_val += 1
                row_count +=1
                print("Validation file count : {}".format(row_val))
                #Read the next line
                line = fr.readline()
            #Write the lines of information for each unique image 
            new_line = ' '.join(chars) + "\n"
            fw.write(new_line)
        print("Total {} lines read with {}/{} lines for training/validation files".format(row_count,row_train,row_val))
        if flags.split==1:
            #Appends data validation txt to training txt by reading the contents
            print("Appending validation data to training data")
            val_in = open(flags.val_file, "r")
            validation_data = val_in.read()
            val_in.close()
            train_in = open(flags.train_file, "a")
            train_in.write(validation_data)
            train_in.close()
            print("Appended with {} lines from validation to training file".format(row_val))



def getBoxCoordinates(array):
    """
    Process the coordinates of the boxes to be within valid range
    """
    Xs = [float(e) for e in array[4:8]]
    Ys = [float(e) for e in array[8:12]]
    tmp_Xmin = min(max(min(Xs), 0), MAX_COORDINATES)
    tmp_Xmax = min(max(max(Xs), 0), MAX_COORDINATES)
    tmp_Ymin = min(max(min(Ys), 0), MAX_COORDINATES)
    tmp_Ymax = min(max(max(Ys), 0), MAX_COORDINATES)
    return str(tmp_Xmin), str(tmp_Xmax), str(tmp_Ymin), str(tmp_Ymax)


if __name__ == "__main__":
    """
    This program generates processed annotations in csv format for training and validation data 
    """
    #Get current directory of this file, which is Vedai
    current_file_dir = os.path.dirname(os.path.realpath("__file__"))

    #DEFINE DEFAULT ARGUMENTS
    DEFAULT_ANNOTATION_INPUT = os.path.join(current_file_dir,"annotations","annotation1024.txt")
    DEFAULT_TRAIN_INPUT = os.path.join(current_file_dir,"annotations","train_vedai.csv")
    DEFAULT_VAL_INPUT = os.path.join(current_file_dir,"annotations","val_vedai.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default = 'CO') #visible (co), ir or visible+ir (coir)
    parser.add_argument("--input_file", default = DEFAULT_ANNOTATION_INPUT) #path to original annotation file
    parser.add_argument("--train_file", default = DEFAULT_TRAIN_INPUT) #output path to training file
    parser.add_argument("--val_file", default = DEFAULT_VAL_INPUT) #output path to validation file
    parser.add_argument("--split", default = 1) #number of samples in training set others will be in validation set
    flags = parser.parse_args()

    if (flags.mode != "IR" and flags.mode != "CO"):
        print("Invalid mode given, exiting the programe...")
        exit(0)

    #Count number of valid files in our dataset by listing the files in the CO/IR folder
    #print(os.listdir(os.path.join(current_file_dir, flags.mode)))
    number_files = len([name for name in os.listdir(os.path.join(current_file_dir, flags.mode)) if os.path.isfile(os.path.join(current_file_dir, flags.mode, name))])
    print("Total number of files in dataset: {}".format(number_files))
    if flags.split<1:
        #Indicate the split factor if <1
        training_files = int(flags.split*number_files)
        print("Split_factor: {}".format(training_files))
    if flags.split==1:
        """
        Setting subset of training files, would enable validation file to be updated with untouched files, which would be appended to the training file
        at the end.
        """
        training_files = int(0.8*number_files)
    training_file_list = getImageFileList(flags.input_file,training_files)
    main(flags,training_file_list)