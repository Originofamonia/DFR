import numpy as np
import math


class NNtrain:
    def __init__(self, XX0delay20test):
        self.Datatrain = np.zeros((18, 53248))
        self.XX1 = XX0delay20test
        self.Datatest = np.zeros((18, 53248))
        self.XXtest = XX0delay20test
        self.Aeta = 1e-3
        self.MaxE = 1e-2
        self.alpha = 0.7

        self.DesireOutput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.randDesireOutput = np.zeros(2)
        self.P = len(self.Datatrain)
        self.NumofInputNeuron = len(self.Datatrain[0])
        self.HiddenLayer1 = 20
        self.HiddenLayer2 = 10
        self.NumofOutputNeuron = 2

        self.v1 = 0.01 * np.random.randn(self.NumofInputNeuron + 1, self.HiddenLayer1)
        self.Dv1 = np.zeros((self.NumofInputNeuron + 1, self.HiddenLayer1))
        self.v2 = 0.01 * np.random.randn(self.HiddenLayer1 + 1, self.HiddenLayer2)
        self.Dv2 = np.zeros((self.HiddenLayer1 + 1, self.HiddenLayer2))
        self.vn = 0.01 * np.random.randn(self.HiddenLayer2 + 1, self.NumofOutputNeuron)
        self.Dvn = np.zeros((self.HiddenLayer2 + 1, self.NumofOutputNeuron))
        self.counter = 0
        self.E = 0
        self.E1 = 0
        self.F = 1
        self.p = 0

        self.ee = np.empty([0])
        self.ee1 = np.empty([0])

    def assign_values(self):
        for i in range(18):
            for j in range(4096):
                self.Datatrain[i, (j * 13): (j * 13 + 12)] = self.XX1[i, j, 0:13]

        for i in range(18):
            for j in range(4096):
                self.Datatest[i, (j * 13): (j*13 + 12)] = self.XXtest[i, j, 0:13]

        for i in range(18):
            self.Datatrain[i, :] = (self.Datatrain[i, :] - np.mean(self.Datatrain[i, :])) \
                                   / np.std(self.Datatrain[i, :])
            self.Datatest[i, :] = (self.Datatest[i, :] - np.mean(self.Datatest[i, :])) \
                                  / np.std(self.Datatest[i, :])

    def sigmoid(self, x):
        if len(x) == 1:
            return 1/ (1 + math.exp(-x))
        else:
            ret = np.array(x)
            for i in range(x):
                ret[i] = 1/ (1 + math.exp(-x[i]))

            return ret

    def train_nn(self):
        while self.F == 1:
            k1 = np.random.permutation(self.P)
            randFace = self.Datatrain
            while self.p <= self.P:
                if 0 <= self.p < 7:
                    self.randDesireOutput = self.DesireOutput[0, :]
                elif 6 < self.p < 13:
                    self.randDesireOutput = self.DesireOutput[1, :]
                elif 12 < self.p < 19:
                    self.randDesireOutput = self.DesireOutput[2, :]
                elif 18 < self.p < 25:
                    self.randDesireOutput = self.DesireOutput[3, :]

                x = randFace[self.p, :]
                y1 = self.sigmoid(np.dot(np.append(x, 1), self.v1))
                y2 = self.sigmoid(np.dot(np.append(y1, 1), self.v2))
                z = self.sigmoid(np.dot(np.append(y2, 1), self.vn))
                deltaZ = self.randDesireOutput - z
                self.E += np.sum(np.power(deltaZ, 2))

                deltaZ = np.dot(np.dot(z, 1-z), deltaZ)
                Y2 = np.append(y2, 1)
                deltaY2 = np.dot(np.dot(Y2, (1-Y2)), deltaZ * np.transpose(self.vn))
                np.delete(deltaY2, self.HiddenLayer2)

                Y1 = np.append(y1, 1)
                deltaY1 = np.dot(np.dot(Y1, (1-Y1)), deltaY2 * np.transpose(self.v2))
                np.delete(deltaY1, self.HiddenLayer1)

                self.Dvn = np.dot(self.Aeta, np.transpose(Y2)) * deltaZ \
                           + np.dot(self.alpha, self.Dvn)
                self.vn = self.vn + self.Dvn

                self.Dv2 = np.dot(self.Aeta, np.transpose(Y1)) * deltaY2 \
                           + np.dot(self.alpha, self.Dv2)
                self.v2 = self.v2 + self.Dv2

                X = np.append(x, 1)
                self.Dv1 = np.dot(self.Aeta, np.transpose(X)) * deltaY1 \
                           + np.dot(self.alpha, self.Dv1)
                self.v1 = self.v1 + self.Dv1

                x11 = self.Datatest[self.p, :]
                y11 = self.sigmoid(np.dot(np.append(x11, 1), self.v1))
                y22 = self.sigmoid(np.dot(np.append(y11, 1), self.v2))
                z11 = self.sigmoid(np.dot(np.append(y22, 1), self.vn))
                deltaz1 = self.randDesireOutput - z11
                self.E1 += np.sum(np.power(deltaz1, 2))
                self.p += 1

            for tt in range(18):
                x1 = self.Datatest[tt, :]
                y11 = self.sigmoid(np.dot(np.append(x1, 1), self.v1))
                y22 = self.sigmoid(np.dot(np.append(y11, 1), self.v2))
                z1 = self.sigmoid(np.dot(np.append(y22, 1), self.vn))

            e1 = math.sqrt(self.E / (self.NumofOutputNeuron * self.P))
            self.ee[self.counter] = e1
            e11 = math.sqrt(self.E1 / (self.NumofOutputNeuron * self.P))
            self.ee1[self.counter] = e11

            print('ee: ', e1)
            print('ee1: ', e11)

            if e1 <= self.MaxE:
                self.F == 0
            else:
                self.E = 0
                self.E1 = 0
                self.p = 0
                self.counter += 1