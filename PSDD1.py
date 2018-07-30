import numpy as np
import math


def PSDD1(TS):
    tsp1 = TS
    t = np.arange(0, 152, 2)
    Ipsc1 = np.zeros((len(TS), len(t)))
    for l in range(len(tsp1)):
        for i in range(len(t)):
            Ipsc1[l,i] = ((math.exp(-(t[i] - tsp1[l]))/10) - math.exp((-t[i]-tsp1[l])/2.5))\
                         * np.heaviside(t[i] - tsp1[l], 0.5)

    Ipsc1 = np.sum(Ipsc1[0:len(TS), :])
    return Ipsc1
