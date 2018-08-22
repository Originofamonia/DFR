import numpy as np
import math


def encoding(J1):

    C = 50 * 1e-2
    Vth = -0.65
    Iext = 120 * 1e-5
    Ileak = 0.32

    a = C * Vth
    A1 = Iext - Ileak

    # Parameters
    N = 3  # must always be >=2, number of intervals
    beta = 3

    # Encoder
    Xi = J1
    D = np.zeros(40)
    for i in range(0, int(math.pow(2, (N - 1) - 1))):
        # case 2*i-1 | 2^(N-1)-1
        D[2*i+1] = Xi * (1/(math.pow(beta, (N-1))))
        # case 2*(2*i-1) | 2^(N-1)-2
        D[2*(2*i+1)] = Xi * (1/math.pow(beta, N-2)) - 1/math.pow(beta, N-1)
        D[int(math.pow(2, N-1)-3)] = Xi * (1/math.pow(beta, N-2)) - 1/math.pow(beta, N-1)
        # case 4*(2*i-1) | 2^(N-1)-4
        D[4*(2*i+1)] = Xi * (1/math.pow(beta, N-3)-1/math.pow(beta, N-2)) - 1/math.pow(beta, N-1)
        # case 8*(2*i-1) | 2^(N-1)-8
        D[8*(2*i+1)] = Xi * (1/math.pow(beta, N-4) - 1/math.pow(beta, N-3) - 1/math.pow(beta, N-2)
                             - 1/math.pow(beta, N-2) - 1/math.pow(beta, N-1))
        if N > 1:
            D[int(math.pow(2, N-1)-2)] = Xi*(1/math.pow(beta, N-2))
        if N > 2:
            D[int(math.pow(2, N-1)-3)] = Xi * (1/math.pow(beta, N-2)-1/math.pow(beta, N-1))
        if N > 3:
            D[int(math.pow(2, N-1)-5)] = Xi*(1/math.pow(beta, N-3)-1/math.pow(beta, N-2)
                                             - 1/math.pow(beta, N-1))
        if N > 4:
            D[int(math.pow(2, N-1)-9)] = Xi*(1/math.pow(beta, N-4) - 1/math.pow(beta, N-3)
                                             - 1/math.pow(beta, N-2) - 1/math.pow(beta, N-1))
        # case 2^(N-2)
        cte2 = 1/beta
        for ibeta in range(1, N-1):
            cte2 = cte2 - 1/math.pow(beta, ibeta)
        if N > 1:
            D[int(math.pow(2, N-2) - 1)] = Xi * cte2
    return D[0: int(math.pow(2, N-1)-1)]
    # return D


def main():
    result = encoding(9)
    print(result)


if __name__ == '__main__':
    main()
