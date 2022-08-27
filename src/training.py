####
# Copyright (c) Abdessalem Mami <abdessalem.mami@esprit.tn>. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
####

import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from models import fer_model
from utils import logger
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import visualkeras


class CFG():
    main_data_dir = "../datasets"
    dataset = "All_Data" 
    dataset_path = os.path.join(main_data_dir,dataset)
    dataset_labels = os.path.join(dataset_path,dataset +"_labels.csv")
    BATCH_SIZE = 32
    IMG_HEIGHT = 48
    IMG_WEIGHT = 48 
    IMG_SIZE = (IMG_HEIGHT,IMG_WEIGHT)
    EPOCHS = 50
    
    def path(subset):
        return os.path.join(CFG.main_data_dir,CFG.dataset, subset)

    

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
    CFG.path("Training"),
    labels='inferred',
    seed=123,
    shuffle=True,
    image_size=CFG.IMG_SIZE,
    label_mode='int',
    #color_mode="grayscale",
    batch_size=CFG.BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    CFG.path("Validation"),
    labels='inferred',
    seed=123,
    shuffle=True,
    image_size=CFG.IMG_SIZE,
    label_mode='int',
    #color_mode="grayscale",
    batch_size=CFG.BATCH_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
    CFG.path("Test"),
    labels='inferred',
    seed=123,
    shuffle=True,
    image_size=CFG.IMG_SIZE,
    label_mode='int',
    #color_mode="grayscale",
    batch_size=CFG.BATCH_SIZE
    )

    return train_ds, val_ds, test_ds


def save_evaluation_plot(history, model_repo_path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = len(history.history['val_loss']) 
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    eval_path = os.path.join(model_repo_path,"history_epochs.png")
    plt.savefig(eval_path)


def save_confusion_matrix(model_repo_path,y_true, y_pred):
    classes=[0,1,2,3,4,5,6]
    cf = confusion_matrix(y_true, y_pred, normalize="all")
    df_cm = pd.DataFrame(cf, index = classes,  columns = classes)
    plt.figure(figsize = (20,20))
    sns.heatmap(df_cm, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    eval_path = os.path.join(model_repo_path,"confusion_matrix.png")
    plt.savefig(eval_path)



def plot_model(model_repo_path, model):
    # Packages issues (https://github.com/XifengGuo/CapsNet-Keras/issues/69)
    # Requires pydot, pydotplus, and graphviz
    model_img = os.path.join(model_repo_path,"model.png")
    tf.keras.utils.plot_model(
    model,
    to_file=model_img,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TR',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True
    )


def start_training(model,train_ds, val_ds, test_ds):
    data = pd.read_csv(CFG.dataset_labels)
    class_weight = dict(zip(range(0, 7), (((data[data['Usage']=='Training']['Emotion'].value_counts()).sort_index())/len(data[data['Usage']=='Training']['Emotion'])).tolist()))
    
    # Early Stopping 
    patience = 5 #Number of epochs with no improvement after which training will be stopped.
    mode = "auto" #The direction is automatically inferred from the name of the monitored quantity.
    restore_best_weights = True #Whether to restore model weights from the epoch with the best value of the monitored quantity.
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode=mode, restore_best_weights=restore_best_weights)
    
    # Train
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG.EPOCHS,
    shuffle=True,
    use_multiprocessing=True,
    class_weight=class_weight, callbacks=[callback])

    # Evaluate on Test Set
    scores = model.evaluate(test_ds, verbose=1)
    print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    
    # Create model directory
    model_repo_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model_repo_path = os.path.join('models', 'history', model_repo_name)
    os.mkdir(model_repo_path)

    # Export model
    model_path = os.path.join(model_repo_path,"fer-model.h5")
    model.save(model_path)

    # Export evaluation history
    save_evaluation_plot(history, model_repo_path)

    # Export confusion matrix 
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    predictions = model.predict(test_ds)
    y_preds = []
    for pred in predictions:
        y_preds.append(np.argmax(pred))

    save_confusion_matrix(model_repo_path, y_true, y_preds)

    # plot model 
    #plot_model(model_repo_path, model)
    model_arch_path = os.path.join(model_repo_path,"architecture.png")
    visualkeras.layered_view(model, to_file=model_arch_path, legend=True)

    # Print model location
    logger.info("Training finished, model saved at " + model_repo_path) 


def train_model():
    logger.info("Training Model...")
    model = fer_model.build_model(input_shape=(CFG.IMG_HEIGHT, CFG.IMG_WEIGHT, 1))
    train_ds, val_ds, test_ds = load_datasets()
    models_zoo = 'models/history/' # Directory where all models and their evaulations are saved
    if not os.path.isdir(models_zoo):
        os.mkdir(models_zoo)
    start_training(model,train_ds, val_ds, test_ds)

