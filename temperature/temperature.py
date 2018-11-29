
import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class Temperature:
    def __init__(self):
        self.n = 125  # number of nodes, n-1 virtual nodes
        self.nv = self.n - 1  # number of virtual nodes = length of mask
        self.k1 = 4 * self.n  # number of initial input data to initialize reservoir
        self.k2 = 4 * self.k1  # number of training input data
        self.k3 = 4 * self.k1  # number of testing input data
        self.k = self.k1 + self.k2 + self.k3  # number of input data
        self.gain = 0.8  # feedback gain
        self.scale = 0.1  # input scaling
        self.m = np.array(self.nv)  # random vector
        self.j = np.zeros(self.k * self.nv)  # masked input?

        self.input = np.array(self.k)
        self.input1 = np.array(self.k1)  # initial input data
        self.input2 = np.array(self.k2)  # training input data
        self.input3 = np.array(self.k3)  # testing input data
        self.target = np.array(self.k)
        self.target1 = np.array(self.k2)  # target output for training
        self.target2 = np.array(self.k3)  # target output for testing
        # Reservoir parameters
        self.x = np.zeros(self.n)  # value in reservoir nodes
        self.x_next = np.zeros(self.n)
        self.X = np.zeros((self.k2, self.n))  # training states
        self.x_all = np.zeros((self.k2 * self.nv, self.n))  # states in all nodes
        self.reg = 1e-8  # regularization coefficient
        self.w_out = np.zeros(self.n)
        self.temp = np.zeros(self.k2)
        self.temp2 = np.zeros(self.k2)

        self.output_train = np.zeros(self.k2)
        self.x_all_test = np.zeros((self.k3 * self.nv, self.n))
        self.j_test = np.zeros(self.k3 * self.nv)

        self.X_test = np.zeros((self.k3, self.n))
        self.output_test = np.zeros(self.k3)

    # Define input & actual output data
    def create_input_output(self):
        arr = np.zeros(10950)
        with open('temp_nl.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                arr[i] = float(row[0])
        self.input = arr[0: 4500]
        self.target = arr[4500: 9000]

        self.input1 = self.input[0: self.k1]
        self.input2 = self.input[self.k1: self.k1 + self.k2]
        self.input3 = self.input[self.k1 + self.k2:]

        self.target1 = self.target[self.k1: self.k1 + self.k2]
        self.target2 = self.target[self.k1 + self.k2:]

    # Defining mask and masking the input
    def mask(self, is_mask=True):
        # j is the masked input
        self.m = np.random.uniform(low=0.0, high=1.0, size=self.nv)
        it = np.nditer(self.input, flags=['f_index'])
        while not it.finished:
            if is_mask:
                masked_elem = np.dot(it[0], self.m)
            else:
                masked_elem = np.repeat(it[0], self.nv)
            for index, value in enumerate(masked_elem):
                self.j[it.index * self.nv + index] = value
            it.iternext()

    # Initialize reservoir
    def init_reservoir(self):
        for i in range(0, self.k1 * self.nv):
            self.x_next[0] = np.tanh(self.scale * self.j[i] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]
            self.x = self.x_next

    # Training data through the reservoir and store the node states.
    def train_reservoir(self):
        for i in range(0, self.k2 * self.nv):
            t = self.k1 * self.nv + i
            self.x_next[0] = np.tanh(self.scale * self.j[t] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]  # in matlab, ] is inclusive!
            self.x = self.x_next
            self.x_all[i, :] = self.x

    # Consider the data just once everytime it loops around?
    def sample_feature(self):
        self.temp = np.arange(0, self.k2, 1)
        self.temp2 = self.nv * self.temp
        self.X[self.temp, :] = self.x_all[self.temp2, :]

    # Train the output weights
    def train_output_weights(self):
        # Here we use regularized least squares.
        numerator = np.dot(self.target1, self.X)
        denominator1 = np.dot(np.transpose(self.X), self.X)
        denominator2 = self.reg * np.identity(self.n)
        denominator = np.add(denominator1, denominator2)
        self.w_out, d1, d3, d4 = np.linalg.lstsq(denominator, numerator, rcond=None)
        return self.w_out, d1, d3, d4

    # Compute training error
    def training_error(self):
        self.output_train = np.dot(self.w_out, np.transpose(self.X))
        error_len = self.k2
        mse = (LA.norm(self.target1 - self.output_train, 2) ** 2) / error_len
        nmse = (LA.norm(self.target1 - self.output_train) / LA.norm(self.target1)) ** 2
        return mse, nmse

    # Testing data through reservoir
    def test(self):
        self.x = np.zeros(self.n)
        self.x_next = np.zeros(self.n)
        self.j_test = self.j[self.nv * (self.k1 + self.k2):len(self.j)]

        # Reservoir initialization
        for i in range(0, self.k1 * self.nv):
            self.x_next[0] = np.tanh(self.scale * self.j[i] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]
            self.x = self.x_next
            self.x_all_test[i, :] = self.x

        # Run data through the reservoir and store the node states.
        for i in range(0, self.k3 * self.nv):
            self.x_next[0] = np.tanh(self.scale * self.j_test[i] + self.gain * self.x[self.n - 1])
            self.x_next[1: self.n] = self.x[0: self.n - 1]
            self.x = self.x_next
            self.x_all_test[i, :] = self.x

        # Consider the data just once everytime it loops around?
        self.temp = np.arange(0, self.k3, 1)
        self.temp2 = self.nv * self.temp
        self.X_test[self.temp, :] = self.x_all_test[self.temp2, :]
        error_len = self.k3
        self.output_test = np.dot(self.w_out, np.transpose(self.X_test))
        mse_test = (LA.norm(self.target2 - self.output_test, 2) ** 2) / error_len
        nmse_test = (LA.norm(self.target2 - self.output_test) / LA.norm(self.target2)) ** 2
        return mse_test, nmse_test

    def draw_charts(self):
        x1 = np.arange(0, 100, 1)
        x2 = np.arange(0, self.k2, 1)
        x41 = np.arange(0, self.k1, 1)
        x42 = np.arange(self.k1, self.k1 + self.k2, 1)
        x43 = np.arange(self.k1 + self.k2, self.k, 1)

        plt.figure(1)
        plt.plot(self.j[self.k1 * self.nv: self.k1 * self.nv + 10 * self.n], marker='x', c=np.random.rand(3, ))
        plt.grid()
        plt.title('Input after Masking')

        plt.figure(2)
        plt.plot(x2, self.output_train, marker='o', c=np.random.rand(3, ))
        plt.plot(x2, self.target1, marker='x', c=np.random.rand(3, ))
        plt.xlabel('o: Reservoir, x: Actual')
        plt.grid()
        plt.title('Train: Reservoir Output vs Actual Output for NARMA-input')

        plt.figure(3)
        plt.plot(x2, self.output_test, marker='o', c=np.random.rand(3, ))
        plt.plot(x2, self.target2, marker='x', c=np.random.rand(3, ))
        plt.xlabel('o: Reservoir, x: Actual')
        plt.grid()
        plt.title('Test: Reservoir Output vs Actual Output for NARMA-input')
        # Below two lines are used to save the results.
        # mat = np.column_stack((x2, self.output_test, self.target2))
        # np.savetxt('narma10_result.csv', mat, delimiter=', ')

        plt.figure(4)
        plt.plot(x41, self.input1, c=np.random.rand(3, ))
        plt.plot(x42, self.input2, c=np.random.rand(3, ))
        plt.plot(x43, self.input3, c=np.random.rand(3, ))
        plt.xlabel('Left: Initial, Middle: Training, Right: Testing')
        plt.grid()
        plt.title('Input Sequence')

        plt.figure(5)
        plt.plot(x41, self.target[0: self.k1], c=np.random.rand(3, ))
        plt.plot(x42, self.target1, c=np.random.rand(3, ))
        plt.plot(x43, self.target2, c=np.random.rand(3, ))
        plt.xlabel('Left: Initial, Middle: Training, Right: Testing')
        plt.grid()
        plt.title('Target Sequence')
        plt.show()


def main():
    t1 = Temperature()
    t1.create_input_output()
    t1.mask(is_mask=True)
    t1.init_reservoir()

    t1.train_reservoir()
    t1.sample_feature()
    t1.train_output_weights()
    mse, nmse = t1.training_error()
    print('mse: ', mse, ', nmse: ', nmse)
    mse_test, nmse_test = t1.test()
    print('mse_test: ', mse_test, ', nmse_test: ', nmse_test)
    t1.draw_charts()


if __name__ == '__main__':
    main()
