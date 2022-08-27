####
# Copyright (c) Abdessalem Mami <abdessalem.mami@esprit.tn>. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
####


import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import mtcnn
import cv2
import tensorflow as tf
from utils import *

# Configuration for DATA PROCESSING
class CFG:
    # Directory path for raw data and also later for processed data after executing the script. 
    main_data_dir = "../datasets"

    # Note: The data preprocessing pipeline expects a labels file for each dataset.
    # The labels file should have columns for the image file name, its emotion label and usage type.

    # Input 
    datasets = {
        "FER":
            {
            "path_to_labels":"FER_labels.csv",
           "processing":{"grayscale":False, "detect_faces":False}
            },
        "IEFD":
            {
            "path_to_labels":"IEFD_labels.csv",
            "processing":{"grayscale":True, "detect_faces":True}
            },
        
    }

    # Output
    new_dataset = "All_Data"
    new_dataset_dir = os.path.join(main_data_dir,new_dataset)
    new_dataset_labels = os.path.join(new_dataset_dir,new_dataset +"_labels.csv")

    # Images Processing Config
    IMG_HEIGHT = 48
    IMG_WEIGHT = 48


def check_inputs():
    logger.info('Checking input directories and files...')
    for dataset in CFG.datasets:
        # Checking if datasets directories exist
        path_to_raw = os.path.join(CFG.main_data_dir, dataset, "raw")
        if not os.path.isdir(path_to_raw):
            logger.error("Path to dataset " + dataset + " doesn't exist.")
            raise SystemExit("")
        # Checking labels 
        path = os.path.join(CFG.main_data_dir, dataset, CFG.datasets[dataset]["path_to_labels"])
        if not os.path.exists(path):
            logger.error("Path to labels of " + dataset + " doesn't exist.")
            raise SystemExit("")
        df = pd.read_csv(path)
        if ("Image" not in df.columns or "Emotion" not in df.columns):
            logger.error(path + " doesn't have Image and Emotion as features")
            raise SystemExit("")



def initialize_directories():
    """
    Create directories for each emotion and for each subset
    """
    logger.info('Initializing new output directory...')
    # Create new dataset directory 
    if not os.path.isdir(CFG.new_dataset_dir):
        os.mkdir(CFG.new_dataset_dir)

    # Create emotions directories
    emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    subsets = ["Training","Validation","Test"]
    for subset in subsets:
        subset_dir = os.path.join(CFG.new_dataset_dir,subset)
        if not os.path.isdir(subset_dir):
            os.mkdir(subset_dir)
            for emotion in emotions:
                emotion_dir = os.path.join(CFG.new_dataset_dir,subset,emotion)
                if not os.path.isdir(emotion_dir):
                    os.mkdir(emotion_dir)




def preprocess_image(face_detector,image_path, target_size=(64,64), face_detection=True):
    img = load_image(image_path)
    # detect face
    if face_detection:
        detected_faces = detect_face(face_detector, img, align = True)
        if len(detected_faces) > 0:
            face, region = detected_faces[0]
            img = face
        #else:
            #logger.warning(image_path + " no face detected")
            #return None


    # Resize image

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    img_pixels = tf.keras.preprocessing.image.img_to_array(img)

    return img_pixels


def emo_mapper(emotion):
    map = {0:"Angry", 1:"Disgusted", 2:"Fearful", 3:"Happy", 4:"Sad", 5:"Surprised", 6:"Neutral"}
    inv_map = {v: k for k, v in map.items()}
    return inv_map[emotion]

def process_dataset(dataset):
    logger.info("Processing images for " + dataset + "...")
    path_to_raw = os.path.join(CFG.main_data_dir, dataset, "raw")
    path_to_labels = os.path.join(CFG.main_data_dir, dataset, CFG.datasets[dataset]["path_to_labels"])
    labels = pd.read_csv(path_to_labels)
    to_detectface = CFG.datasets[dataset]["processing"]["detect_faces"]
    data = {}
    face_detector = mtcnn.MTCNN()
    for index, row in labels.iterrows():
        filename = row["Image"]
        emotion = row["Emotion"]
        usage = row["Usage"]
        image_path = os.path.join(path_to_raw,filename)
        if os.path.exists(image_path):
            image = preprocess_image(face_detector,image_path, target_size=(CFG.IMG_HEIGHT,CFG.IMG_WEIGHT), face_detection=to_detectface)
            if image is not None:
                destination_path = os.path.join(CFG.new_dataset_dir, usage, emotion, filename)
                tf.keras.utils.save_img(destination_path, image)
                data[index] = {"Dataset":dataset, "Usage":usage,"Image":filename, "Emotion":emotion}
        else:
            logger.warning(image_path + "Image doesnt exist")
    return data


def preprocess_datasets():
    if not os.path.exists(CFG.new_dataset_labels):
        logger.info("Processing datasets...")
        check_inputs()
        initialize_directories()
        ldata = []
        for dataset in CFG.datasets:
            data = process_dataset(dataset)
            ldata.append(pd.DataFrame(data).transpose())
        df = pd.concat(ldata, axis=0).reset_index(drop=True)
        df.to_csv(CFG.new_dataset_labels)



