import keras
import keras.callbacks as cbks
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Model
from keras.engine.training import _make_batches
from datetime import datetime as dt
from tqdm import tqdm
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import os, sys, random
from datetime import datetime as dt

def calc_steps_per_epoch(index_list, batch_size=os.environ["BATCH_SIZE"]):
    steps = math.ceil(len(index_list) / int(batch_size))
    return steps

# コールバッククラスの有効化
# callback lists :
#   TensorBoard
# arg1 log_dir : TensorBoardのlogファイルを保存するディレクトリ
def activate_callbacks(log_dir=None):
    if log_dir == None:
        raise ValueError("Invalid argument : log_dir is None")
    elif len(log_dir) == 0:
        raise ValueError("Invalid argument : log_dir is empty")
    elif isinstance(log_dir, str) is not True:
        raise ValueError("Invalid argument : log_dir is not string object")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)
    early_stopping = EarlyStopping(monitor="val_loss", verbose=1, mode="min")
    callbacks = [tensorboard]
    return callbacks

# 学習曲線をグラフに出力する
# arg1 history : train, train_generator等の返り値のKeras.Historyオブジェクト
def write_epoch_curve(model_name, history):
    running_datetime_str = dt.now().strftime("%Y%m%d%H%M")
    loss  = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epoch = len(loss)
    fig, axis1 = plt.subplots()
    axis1.plot(range(epoch), loss, marker='.', color="b", label="loss")
    axis1.plot(range(epoch), val_loss, marker='.', color="r", label="validation loss")
    axis2 = axis1.twinx()
    axis2.plot(range(epoch), acc, marker='.', color="g", label="accuracy")
    axis2.plot(range(epoch), val_acc, marker='.', color="c", label="validation accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("./results/train/%s_%s.png"%(model_name, running_datetime_str))
    return "./results/train/%s_%s.png"%(model_name, running_datetime_str)
