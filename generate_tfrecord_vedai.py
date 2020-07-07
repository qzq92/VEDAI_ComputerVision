import os
import numpy as np
import io
import pandas as pd
import tensorflow as tf
import hashlib
import argparse
from PIL import Image

from pathlib import Path
import sys

current_file_dir = os.path.dirname(os.path.realpath("__file__"))
path = Path(current_file_dir)

#Add tensorflow models/research folder for external imports
tf_models = os.path.join(path.parent,"models","research")
slim_dir = os.path.join(path.parent,"models","research","slim")
objectdetect_dir = os.path.join(path.parent,"models","research","object_detection")
local_bin = os.path.join("/usr","local","bin")

print("Adding tensorflow models: {}, {}, {} to sys path...".format(tf_models,slim_dir,local_bin))
sys.path.insert(0,tf_models)
sys.path.insert(0,slim_dir)
sys.path.insert(0,objectdetect_dir)
sys.path.insert(0,local_bin)

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

Image.MAX_IMAGE_PIXELS = 1000000000

# Label mapping
# 1: car, 2:trucks, 4: tractors, 5: camping cars, 7: motorcycles, 8:buses, 9: vans, 10: others, 11: pickup, 23: boats , 201: Small Land Vehicles, 31: Large land Vehicles
def class_text_to_int(col_label):
    if col_label == 'car':
        return 1
    elif col_label == 'trucks':
        return 2
    elif col_label == 'tractors':
        return 4
    elif col_label == 'camping cars':
        return 5
    elif col_label == 'motorcycles':
        return 7
    elif col_label == 'buses':
        return 8
    elif col_label == 'vans':
        return 9
    elif col_label == 'others':
        return 10
    elif col_label == 'pickup':
        return 11
    elif col_label == 'boats':
        return 23
    elif col_label == 'Large land vehicles':
        return 31
    elif col_label == 'Small land vehicles':
        return 201
    else:
        return None

def int_to_class_label(integer_class):
    #returns byte encoding of each class
    if integer_class == 1:
        return b"car"
    elif integer_class == 2:
        return b"trucks"
    elif integer_class == 4:
        return b"tractors"
    elif integer_class == 5:
        return b"camping cars"
    elif integer_class == 7:
        return b"motorcycles"
    elif integer_class == 8:
        return b"buses"
    elif integer_class == 9:
        return b"vans"
    elif integer_class == 10:
        return b"others"
    elif integer_class == 11:
        return b"pickup"
    elif integer_class == 23:
        return b"boats"
    elif integer_class == 31:
        return b"Large land vehicles"
    elif integer_class == 201:
        return b"Small land vehicles"
    else:
        return b"Unknown"


def create_tf_example(annotation_dict_key,annotation_dict_value, image_path):
    """
    Generates a tf Example for each image represented by dictionary key and the
    bounding boxes and labels represented by annotation_dict_value, with the image path
    """
    #print(annotation_dict)
    image_format = b'png'  #change to jpg or jpeg if required
    classes = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    #Read the images from the corresponding folder based on type
    bboxes_class= list(annotation_dict_value.split(" "))
    if len(bboxes_class)%5==0:
        #Exclude the annotated data and image if the length is incorrect
        print(bboxes_class)
        with tf.io.gfile.GFile(os.path.join(image_path, annotation_dict_key), 'rb') as fid:
            encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        key = hashlib.sha256(encoded_png).hexdigest().encode('utf8')
        image = Image.open(encoded_png_io)
        width, height = image.size
        filename = annotation_dict_key.encode('utf8')

        for i in range(len(bboxes_class)):
            if i%5==0:
                classes.append(int(bboxes_class[i]))
            elif i%5==1:
                xmins.append(np.float32(bboxes_class[i]) / width)
            elif i%5==2:
                ymins.append(np.float32(bboxes_class[i]) / width)
            elif i%5==3:
                xmaxs.append(np.float32(bboxes_class[i]) / width)
            else:
                ymaxs.append(np.float32(bboxes_class[i]) / width)

        classes_text = list(map(int_to_class_label, classes))
        #print(classes)
        #print(classes_text)
            #Extract bounding boxes and its labels and the corresponding file information as a tensorflow example
            #Fundamentally, a tf.Example is a {"string": tf.train.Feature} mapping.
        feature_dict ={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/key/sha256':dataset_util.bytes_feature(key) #Generate your own
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return tf_example

def csv_to_dict(csv_data):
    annotation_df= csv_data[0].str.split(" ", n = 1, expand = True)
    #print(annotation_df.head())
    #Rename the columns and set the image id as index
    annotation_df.rename(columns={0: 'image', 1: 'annotation'}, inplace=True)
    annotation_df.set_index("image",inplace=True)
    print(annotation_df.head())
    annotation_dict = annotation_df["annotation"].to_dict()

    return annotation_dict

def main(_):
    #Setup TFrecordwriter to write to output path
    writer_train = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path_train)

    #Get training image directory
    train_image_path = FLAGS.train_image_dir

    #train_image_path = os.path.join(os.getcwd(), "train")
    #validation_image_path = os.path.join(os.getcwd(), "validation")
    #Read csv files without headers
    train_examples = pd.read_csv(FLAGS.train_annotations_file,header=None)
    train_dict = csv_to_dict(train_examples)
    print("Creating tf examples for training example")
    for train_dict_k,train_dict_v in train_dict.items():
        tf_example = create_tf_example(train_dict_k,train_dict_v,FLAGS.train_image_dir)
        writer_train.write(tf_example.SerializeToString())

    writer_train.close()
    output_training = os.path.join(os.getcwd(), FLAGS.output_path_train)
    print('Successfully created the TFRecords for Vedai training dataset: {}'.format(output_training))

    writer_val = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path_val)
    #Get validation image directory
    validation_image_path = FLAGS.val_image_dir
    print("Creating tf examples for validation example")
    validation_examples = pd.read_csv(FLAGS.val_annotations_file,header=None)
    val_dict = csv_to_dict(validation_examples)
    for val_dict_k,val_dict_v in val_dict.items():
        tf_example = create_tf_example(val_dict_k,val_dict_v,FLAGS.val_image_dir)
        writer_val.write(tf_example.SerializeToString())
    #Serialize Data ToString
    #Close writer
    writer_val.close()
    output_validation = os.path.join(os.getcwd(), FLAGS.output_path_val)
    print('Successfully created the TFRecords for Vedai training dataset: {}'.format(output_validation))


if __name__ == '__main__':
    """
    This program generates TFRecored format for object detection required for VEDAI dataset
    Usage:
    # From tensorflow/models/
    # Create train data:
    python generate_tfrecord.py --csv_input=annotation/train.csv  --output_path=annotation/train.record --type=train
    # Create test data:
    python generate_tfrecord.py --csv_input=annotation/val.csv  --output_path=annotation/test.record --type=test
    """
    #Get current directory of this file, which is Vedai
    current_file_dir = os.path.dirname(os.path.realpath("__file__"))
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", default = '') #Path to the training images
    parser.add_argument("--val_image_dir", default = '') #Path to the validation images
    parser.add_argument("--train_annotations_file", default = '') #Path to the training annotations
    parser.add_argument("--val_annotations_file", default = '') #Path to the validation annotations
    parser.add_argument("--output_path_train", default = '') #Path to output TFRecord for training data
    parser.add_argument("--output_path_val", default = '') #Path to output TFRecord for validation data
    parser.add_argument("--type", default = 'CO') #CO or IR
    flags = parser.parse_args()
    """
    #Define tensorflow flags    
    flags = tf.compat.v1.app.flags
    flags.DEFINE_string('train_image_dir', '', 'Path to the training images')
    flags.DEFINE_string('val_image_dir', '', 'Path to the validation images')
    flags.DEFINE_string('train_annotations_file', '', 'Path to the training annotations')
    flags.DEFINE_string('val_annotations_file', '', 'Path to the validation annotations')
    flags.DEFINE_string('output_path_train', '', 'Path to output TFRecord for training data')
    flags.DEFINE_string('output_path_val', '', 'Path to output TFRecord for validation data')
    flags.DEFINE_string('type', '', 'CO or IR')
    FLAGS = flags.FLAGS
    #Runs the main function with no optional argument. flags is defined for you automatically
    tf.compat.v1.app.run(main=main,argv=None)
    #tf.app.run()
