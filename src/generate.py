####
# Copyright (c) Abdessalem Mami <abdessalem.mami@esprit.tn>. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
####

import os
import wget
import shutil
import pandas as pd 
from PIL import Image
import numpy as np 
from utils import logger, bar_custom, str_to_image

data_directory = "../datasets/"
emotion_table = {'neutral': 6, 'Neutral': 6, 'happiness': 3, 'happy': 3,'Happiness':3,'surprise' : 5, 'Surprise' : 5, 
'sadness': 4, 'Sadness':4,'sad': 4, 'anger': 0, 'Anger':0,'angry': 0, 'disgust': 1, 'Disgust'  : 1, 'fear': 2, 'Fear': 2}
map2 = {0:"Angry", 1:"Disgusted", 2:"Fearful", 3:"Happy", 4:"Sad", 5:"Surprised", 6:"Neutral"}

def str_to_image(image_blob):
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return Image.fromarray(image_data)


def download_datasets():
    data_path = data_directory + "Original.zip" 
    label_path = data_directory + "IEFD/ImageDescriptionWEmotions.csv"
    data_urls = {
    "IEFD":"https://osf.io/download/nf2by/",
    "IEFD_labels":"https://osf.io/download/b8gzm/",
    #"FER":"", 
    }

    # Download IEFD images
    if not os.path.exists(data_path):
        logger.info('Downloading IEFD files...')
        wget.download(data_urls["IEFD"],data_path, bar=bar_custom)

    # Download IEFD labels
    if not os.path.exists(label_path):
        logger.info('Downloading IEFD labels...')
        wget.download(data_urls["IEFD_labels"],label_path , bar=bar_custom)



def initialize_directories():
    logger.info('Initializing directories...')
    directories = ["FER","IEFD"]
    for directory in directories:
        dir_path = data_directory + directory
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        


# Generating FER+ 2013 
def generate_FER():
    # check if FER csv exists
    if not os.path.exists(data_directory + "icml_face_data.csv"):
        logger.error("The file icml_face_data.csv is missing.")
        logger.error("Please download it from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv")
        logger.error("Don't forget to move it into /datasets directory")
        raise SystemExit("")
    
    # check if FER+ file exists
    if not os.path.exists(data_directory + "fer2013new.csv"):
        logger.error("The file fer2013new.csv is missing.")
        raise SystemExit("")

    # check if files are not already generated
    if not os.path.isdir(data_directory + "FER/raw"):
        logger.info("Generating FER+ 2013 files...")
        os.mkdir(data_directory + "FER/raw")
        # load csv files 
        fer = pd.read_csv(data_directory + "icml_face_data.csv", usecols=[" pixels"])
        ferplus = pd.read_csv(data_directory + "fer2013new.csv")
        emo_cols = ferplus.columns.to_list()[2:]
        data = []
        umap = {"Training":"Training", "PublicTest":"Validation", "PrivateTest":"Test"}
        for i, row in ferplus.iterrows():
            emotions = np.array(row.loc[emo_cols])
            dom_emotion = emo_cols[emotions.argmax()]
            # skip contempt, NF, unknown 
            if dom_emotion in ["contempt","unknown","NF"]:
                continue
            
            # save image and label
            image_pixels = fer.iloc[i][0]
            image_name = row["Image name"]
            usage = row["Usage"]
            data.append({"Usage":umap[usage],"Image":image_name,"Emotion":dom_emotion})
            image = str_to_image(image_pixels)
            image_path = os.path.join(data_directory + "FER","raw", image_name)
            image.save(image_path, compress_level=0, bitmap_format="png")   


        corrected_fer = pd.DataFrame(data)
        corrected_fer["Emotion"] = corrected_fer["Emotion"].map(emotion_table)
        corrected_fer["Emotion"] = corrected_fer["Emotion"].map(map2)
        corrected_fer.to_csv(data_directory + "FER/FER_labels.csv")



def generate_IEFD():
    if not os.path.isdir(data_directory + "IEFD/raw"):
        logger.info("Extracting IEFD files...")
        shutil.unpack_archive(data_directory + 'Original.zip', data_directory + 'IEFD/')
        os.rename(data_directory + "IEFD/Original", data_directory + "IEFD/raw")
    if not os.path.exists(data_directory + "IEFD/ImageDescriptionWEmotions.csv"):
        logger.error("ImageDescriptionWEmotions.csv is missing. Make sure it was downloaded.")
        raise SystemExit("")
    
    iefd_labels = pd.read_csv(data_directory + "IEFD/ImageDescriptionWEmotions.csv", usecols=["PicNum","IntendedExpression"])
    iefd_labels = iefd_labels.rename(columns={"PicNum":"Image","IntendedExpression":"Emotion"})
    iefd_labels["Image"] = iefd_labels["Image"].astype(str) + ".jpg"

    iefd_labels["Emotion"] = iefd_labels["Emotion"].map(emotion_table)
    iefd_labels["Emotion"] = iefd_labels["Emotion"].map(map2)
    iefd_labels = iefd_labels.assign(Usage="Validation")  # All IEFD dataset will be used for Validation

    iefd_labels.to_csv(data_directory + "IEFD/IEFD_labels.csv",index=False)
    

def generate_datasets():
    logger.info('Downloading and generating datasets...')
    # init and create required directories 
    initialize_directories()

    # generate FER images from csv file 
    generate_FER()

    # This will ONLY automatically download IEFD dataset if it doesn't exist
    # For FER dataset, you should manually download icml_face_data.csv
    download_datasets()

    # extract IEFD images and unify labels
    generate_IEFD()


    