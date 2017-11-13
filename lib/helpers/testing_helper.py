import keras
from keras.models import Model
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
import matplotlib.pyplot as plt
import argparse
import os, sys
import random

def draw_confusion_matrix(matrix=None, labels=None, model_name=None):
    if matrix is None:
        raise ValueError("Invalid argument : matrix is None")
    if labels is None:
        labels = range(len(matrix))
    if model_name is None:
        raise ValueError("Invalid argument : model_name is None")
    elif isinstance(model_name, str) is not True:
        raise ValueError("Invalid argument : model_name is not string object")
    elif len(model_name) is 0:
        raise ValueError("Invalid argument : model_name is empty")
    datetime_str = dt.now().strftime("%Y%m%d%H%M")
    fig = plt.figure()
    axis = fig.add_subplot(111)
    confusion_matrix = axis.matshow(matrix, cmap='jet')
    fig.colorbar(confusion_matrix)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    plt.ylabel("Correct index")
    plt.xlabel("Predicted index")
    plt.savefig("./results/test/%s_%s.png"%(model_name, datetime_str))

def predict(model=None, index_list=None, class_num=10, generator=None):
    confusion_matrix = np.zeros((class_num, class_num))
    for i, batch in enumerate(tqdm(generator.generator(self=generator,
                                                       index_list=index_list,
                                                       batch_size=1))):
        if len(index_list) < i:
            break
        predict_x , index = batch
        result = model.predict_on_batch(predict_x)
        result_index = np.argmax(result)
        confusion_matrix[np.argmax(index)][result_index] += 1

    confusion_matrix = confusion_matrix / sum(confusion_matrix)
    return confusion_matrix

def calc_accuracy(confusion_matrix=None):
    accuracy = 0
    precisions = [0 for i in range(len(confusion_matrix))]
    for i in range(len(confusion_matrix)):
        accuracy += confusion_matrix[i][i]
        precision = confusion_matrix[i][i] / (sum(confusion_matrix[:,i]) + 1)
        precisions[i] = precision
    accuracy = accuracy / np.sum(confusion_matrix)
    return accuracy, precisions
