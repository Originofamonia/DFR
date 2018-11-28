from scipy.io import loadmat
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class ECG:
    def __init__(self):
        mat = loadmat('./100m.mat')
        self.train_data = mat['val']


def main():
    mat = loadmat('./100m.mat')
    val = mat['val']
    print(val)


if __name__ == '__main__':
    main()


'''
cpu + mobo + ssd: 500
cpu cooler: 75
thermal paste: 10
gpu 2080: 650
ram: 120
case: 45
power supply: 0 already bought with 1060 (250)
total:  
'''