import numpy as np
import LIF1
import PSDD1


class Reservoir(object):

    def __init__(self, ipsc):
        self.Data = ipsc
        self.gain = 0.8
        self.XX = np.zeros((18, 4096, 13))
        self.X1 = np.zeros((18, 4096, 13))
        self.X2 = np.zeros((18, 4096, 13))
        self.X3 = np.zeros((18, 4096, 13))
        self.X4 = np.zeros((18, 4096, 13))
        self.rate = np.zeros((4096, 4096))
        self.SPIKE = np.zeros((76, 76))
        # self.TT = np.zeros(6)
        self.Inew = np.zeros((18, 4096, 76))

    def assign_values(self):
        for i in range(4096):
            for j in range(6):
                if j == 0:
                    self.SPIKE, TT = LIF1.LIF1(16 * self.Data[0, i, :])
                    self.XX[0, i, 0:len(TT)] = TT
                    self.X1[0, i, 0:len(TT)] = TT + 20
                    self.X2[0, i, 0:len(TT)] = TT + 40
                    self.X3[0, i, 0:len(TT)] = TT + 60
                    self.X4[0, i, 0:len(TT)] = TT + 80
                    self.Inew[0, i, :] = PSDD1.PSDD1(TT)
                    self.rate[i, j] = np.count_nonzero(TT)
                else:
                    self.SPIKE, TT = LIF1.LIF1(16 * self.Data[j, i, :]
                                               + 0.8 * self.Inew[j-1, i, :])
                    self.XX[j, i, 0:len(TT)] = TT
                    self.X1[j, i, 0:len(TT)] = TT + 20
                    self.X2[j, i, 0:len(TT)] = TT + 40
                    self.X3[j, i, 0:len(TT)] = TT + 60
                    self.X4[j, i, 0:len(TT)] = TT + 80
                    self.Inew[j, i, :] = PSDD1.PSDD1(self.X4[j, i, 0: len(TT)])
                    self.rate[i, j] = np.count_nonzero(TT)

            for j in range(6, 12):
                if j == 6:
                    self.SPIKE, TT = LIF1.LIF1(16* self.Data[6, i, :])
                    self.XX[6, i, 0:len(TT)] = TT
                    self.X1[6, i, 0:len(TT)] = TT + 20
                    self.X2[6, i, 0:len(TT)] = TT + 40
                    self.X3[6, i, 0:len(TT)] = TT + 60
                    self.X4[6, i, 0:len(TT)] = TT + 80
                    self.Inew[6, i, :] = PSDD1.PSDD1(TT)
                    self.rate[i, j] = np.count_nonzero(TT)
                else:
                    self.SPIKE, TT = LIF1.LIF1(16 * self.Data[j, i, :]
                                               + 0.8 * self.Inew[j - 1, i, :])
                    self.XX[j, i, 0:len(TT)] = TT
                    self.X1[j, i, 0:len(TT)] = TT + 20
                    self.X2[j, i, 0:len(TT)] = TT + 40
                    self.X3[j, i, 0:len(TT)] = TT + 60
                    self.X4[j, i, 0:len(TT)] = TT + 80
                    self.Inew[j, i, :] = PSDD1.PSDD1(self.X4[j, i, 0: len(TT)])
                    self.rate[i, j] = np.count_nonzero(TT)

            for j in range(12, 18):
                if j == 12:
                    self.SPIKE, TT = LIF1.LIF1(16 * self.Data[12, i, :])
                    self.XX[12, i, 0:len(TT)] = TT
                    self.X1[12, i, 0:len(TT)] = TT + 20
                    self.X2[12, i, 0:len(TT)] = TT + 40
                    self.X3[12, i, 0:len(TT)] = TT + 60
                    self.X4[12, i, 0:len(TT)] = TT + 80
                    self.Inew[12, i, :] = PSDD1.PSDD1(TT)
                    self.rate[i, j] = np.count_nonzero(TT)
                else:
                    self.SPIKE, TT = LIF1.LIF1(16 * self.Data[j, i, :]
                                               + 0.8 * self.Inew[j - 1, i, :])
                    self.XX[j, i, 0:len(TT)] = TT
                    self.X1[j, i, 0:len(TT)] = TT + 20
                    self.X2[j, i, 0:len(TT)] = TT + 40
                    self.X3[j, i, 0:len(TT)] = TT + 60
                    self.X4[j, i, 0:len(TT)] = TT + 80
                    self.Inew[j, i, :] = PSDD1.PSDD1(self.X4[j, i, 0: len(TT)])
                    self.rate[i, j] = np.count_nonzero(TT)

        return self.XX, self.rate
