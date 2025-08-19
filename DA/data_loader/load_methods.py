import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat


def CWRU(item_path):
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = os.path.basename(item_path).split(".")[0]
    if eval(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]
    signal = loadmat(item_path)[realaxis]

    return signal


def MFPT(item_path):
    f = item_path.split("/")[-2]
    if f == 'normal':
        signal = (loadmat(item_path)["bearing"][0][0][1])
    else:
        signal = (loadmat(item_path)["bearing"][0][0][2])

    return signal


def PU(item_path):
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    signal = fl[0][0][2][0][6][2]  #Take out the data
    signal = signal.reshape(-1,1)

    return signal


def XJTU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl["Horizontal_vibration_signals"]
    signal = signal.values.reshape(-1,1)

    return signal


def IMS(item_path):
    channel = {'normal': 0,
               'inner': 4,
               'outer': 0,
               'ball': 6}
    f = item_path.split("/")[-2]
    signal = np.loadtxt(item_path)[:, channel[f]]

    return signal


def JNU(item_path):
    fl = pd.read_csv(item_path)
    signal = fl.values
    
    return signal

def source1kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    # print(signal.shape)
    return signal

def target1kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source1unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def target1unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source2kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    # print(signal.shape)
    return signal

def target2kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source2unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def target2unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source3kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    # print(signal.shape)
    return signal

def target3kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source3unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def target3unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source4unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def target4unk(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def source4kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal

def target4kn(item_path):
    img = Image.open(item_path).convert('L')
    signal = np.array(img)
    return signal



