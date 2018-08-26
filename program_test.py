# To test some functions

import numpy as np


def function1():
    outfile = 'file.npy'
    x = np.arange(9)
    np.save(outfile, x)

    x2 = np.load(outfile)
    return x2


def function2():
    x = np.arange(9)
    x2 = 16 * x
    return x2


def main():
    ans = function2()
    print(ans)


if __name__ == '__main__':
    main()

