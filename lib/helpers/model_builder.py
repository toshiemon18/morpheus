import keras
from keras.models import Model
from datetime import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import h5py
import numpy as np
import os, sys, random, re

# h5ファイルから学習済みパラメータを読み込む
# 引数 :
#   model_name   : 学習モデルの名前(ファイル名)
#   dataset_name : データセット名
def load_weights(model_name=None, dataset_name=None):
    if model_name is None or dataset_name is None:
        raise ValueError("Invalid argument : model_name and dataset_name is required")
    weights_path = "./weights/%s/"%dataset_name
    weights = os.listdir(weights_path)
    weights = [w for w in weights if re.match(r'%s_'%model_name, w)]
    weights.sort()
    # weight = h5py.file(weights_path + weights[0])
    # return weight

# 学習済みパラメータをモデルに適用する
# 引数 :
#   model       : 使用する学習モデル
#   removal_num : 除去するレイヤーの数 (default=0, bottom-up)
#   weights     : 学習済みパラメータのh5pyオブジェクト
def apply_weights(model, removal_num=0, weights=None):
    if weights == None:
        raise ValueError("Invalid argument : weights is None, this parameter is required")
    if removal_num < 0:
        raise Exception("Invalid argument : removal_num must not be a negative value")
    elif type(removal_num) is not int:
        raise Exception("Invalid argument : set a integer value to removal_num")
    if model == None:
        raise Exception("Invalid argument : model is None, this parameter is required")
    weights_value_tuples = []
    layers = [l.name for l in model.layers]
    # 出力層から数えてremoval_num層ぶんだけリストから削除
    del(layers[-removal_num:])
    for n, name in enumerate(layers):
        g = weights[name]
        weight_name = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weigt_name in weight_names]
            layer = model.layers[n]
            symbolic_weights = layer.trianable_weights + layer.non_trainable_weights

        if len(weight_values) != len(symbolic_weights):
            raise Exception('Layer #' + str(k) +
                            ' (named "' + layer.name +
                            '" in the current model) was found to '
                            'correspond to layer ' + name +
                            ' in the save file. '
                            'However the new layer ' + layer.name +
                            ' expects ' + str(len(symbolic_weights)) +
                            ' weights, but the saved weights have ' +
                            str(len(weight_values)) +
                            ' elements.')
        weights_value_tuples += zip(symbolic_weights, weight_values)
    keras.backend.batch_set_value(weights_value_tuples)
    return model

# パラメータを更新しない層を設定する
# 引数 :
#   model              : KerasのModelオブジェクト
#   untrainable_layers : 更新しない層数(default=0, top-down)
def set_untrainable_layers(model=None, untrainable_layers=0):
    layers = [l for l in model.layers]
    for layer, n in enumerate(layers):
        if (n+1) < untrainable_layers:
            layer.trainable = False
        else:
            layer.trainable = True
    return model
