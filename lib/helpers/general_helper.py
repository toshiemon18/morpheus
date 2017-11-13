import keras
from datetime import datetime as dt
from tqdm import tqdm
import h5py
import numpy as np
import math
import os, sys, random
import csv
import pandas as pd

# 各データセットのディレクトリ構造
#   + PREFIX_DATASET_DIR/
#       + train.csv         学習用のデータリスト(csv形式)
#       + test.csv          テスト用のデータリスト(csv形式)
#       + data/
#           + [0-9]*.npy    NSGTで変換した入力データ(numpy array形式)
def fetch_datalist(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Invalid argument : dataset_name. dataset_name is None.")
    if os.path.exists("%s/%s"%(os.environ["PREFIX_DATASET_DIR"], dataset_name)):
        raise Exception("No such directory : %s/%s is not found."%(os.environ["PREFIX_DATASET_DIR"], dataset_name))
    # CSVファイルをopen
    prefix = os.environ["PREFIX_DATASET_DIR"]
    test_datalist = open(prefix + os.environ[dataset_name] + "/test.csv")
    train_datalist = pd.read_csv(prefix + "/" + os.environ[dataset_name] + "/train.csv")
    train_datalist = list(train_datalist.values.tolist())
    test_datalist =  pd.read_csv(prefix + "/" + os.environ[dataset_name] + "/test.csv")
    test_datalist =  list(test_datalist.values.tolist())
    return train_datalist, test_datalist

