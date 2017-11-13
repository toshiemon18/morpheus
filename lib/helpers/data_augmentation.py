import numpy as np
import random
import os

# スペクトラムにホワイトノイズを加える
# === Args
#   [spectrum]     array like  NSGTによるスペクトラム
# === Return
#   [additive_whitenoise] array like  ホワイトノイズを加えたスペクトラム
def additive_whitenoise(spectrum):
    shapes = spectrum.shape
    index = random.randrange(0, 10) % 3
    # 0か1のときノイズを加える
    # 2なら添加せずそのまま出力
    additive_spectrum = None
    if index in [0, 1]:
        whitenoise = np.load("%s/noise/noise%s.npy"%(os.environ["PREFIX_DATASET_DIR"], index))
        additive_spectrum = np.add(spectrum, whitenoise)
    else:
        additive_spectrum = spectrum
    return additive_spectrum
