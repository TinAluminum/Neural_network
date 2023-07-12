import random as rd
import numpy as np
import matplotlib.pyplot as plt
# from titanic_data_processing import Datanp as X
# from titanic_data_processing import Survivednp as Y
# from training_data import X, Y, m
from time import sleep


class NN:
    def __init__(self, architecture):
        self.arch = architecture
        weights = {}
        for index in range(len(architecture) - 1):
            layer_number = index + 1
            current_layer_size = architecture[index]
            next_layer_size = architecture[index + 1]
            if index != len(architecture) - 1:
                current_layer_size += 1
            layer_theta = np.random.uniform(-0.5, 0.5, (current_layer_size, next_layer_size))
            key = str(layer_number) + '-' + str(layer_number + 1)
            weights[key] = layer_theta
        deltas = {}
        for key in weights.keys():
            deltas[key] = np.zeros(weights[key].shape)
        self.weights = weights
        self.Deltas = deltas
        self.fpd = {}

    def forward_pass(self, X, m):
        pass_num = len(self.arch) - 1
        for passes in range(pass_num):
            key = str(passes + 1) + '-' + str(passes + 2)
            fpd_zkey = 'z' + str(passes + 1);
            fpd_akey = 'a' + str(passes + 1)
            if passes == 0:
                current_mat = X
            else:
                current_mat = A
            Theta = self.weights[key]
            Z = np.matmul(Theta.T, current_mat)
            A_no_bias = sigmoid(Z)
            if passes == pass_num - 1:
                self.fpd[fpd_zkey] = Z;
                self.fpd['h'] = A_no_bias
                break
            bias = np.ones((1, m))
            A = np.append(bias, A_no_bias, axis=0)
            self.fpd[fpd_zkey] = Z;
            self.fpd[fpd_akey] = A

        pass

    def back_propagation(self, X, Y, m):
        # h: 1x4
        # X: 3x4, layer_len+bias, m
        deltas = {}
        total_layer = len(self.arch)
        for training_set in range(m):
            ans = Y[:, training_set].reshape((self.arch[-1], 1))

            hypos = self.fpd['h'][:, training_set]

            d_highest = hypos - ans

            deltas[total_layer] = d_highest
            O = True
            passes_left = len(self.arch) - 2
            while O == True:
                current_layer = passes_left + 2;
                lower_layer = passes_left + 1
                theta_key = str(lower_layer) + '-' + str(current_layer)
                if passes_left != 0:
                    Theta = self.weights[theta_key][1:, :];
                    delta = deltas[passes_left + 2]
                    step1 = np.matmul(Theta, delta)
                    ########
                    zkey = 'z' + str(passes_left)
                    Z = self.fpd[zkey][:, training_set]
                    Z = Z.reshape((len(Z), 1))
                    d = np.multiply(step1, sigmoid_gradient(Z))
                    ########

                    deltas[lower_layer] = d
                    passes_left -= 1


                else:

                    for i in range(len(self.Deltas.keys())):
                        key = str(i + 1) + '-' + str(i + 2)
                        akey = 'a' + str(i)
                        if akey not in self.fpd.keys():
                            step1 = X[:, training_set].reshape((self.arch[0] + 1, 1))
                        else:
                            step1 = self.fpd[akey][:, training_set]
                        step2 = step1.reshape((len(step1), 1))
                        self.Deltas[key] += np.matmul(deltas[i + 2], step2.T).T
                    break
        for layer in range(len(self.arch) - 1):
            key = str(layer + 1) + '-' + str(layer + 2)
            self.Deltas[key] /= m

    def update(self, learning_rate):
        for layer in range(len(self.arch) - 1):
            key = str(layer + 1) + '-' + str(layer + 2)
            self.weights[key] -= learning_rate * self.Deltas[key]

    def forward_input(self, X, m):
        pass_num = len(self.arch) - 1
        for passes in range(pass_num):
            key = str(passes + 1) + '-' + str(passes + 2)
            fpd_zkey = 'z' + str(passes + 1);
            fpd_akey = 'a' + str(passes + 1)
            if passes == 0:
                current_mat = X
            else:
                current_mat = A
            Theta = self.weights[key]
            Z = np.matmul(Theta.T, current_mat)
            A_no_bias = sigmoid(Z)
            if passes == pass_num - 1:
                self.fpd[fpd_zkey] = Z;
                self.fpd['h'] = A_no_bias
                break
            bias = np.ones((1, m))
            A = np.append(bias, A_no_bias, axis=0)
            self.fpd[fpd_zkey] = Z;
            self.fpd[fpd_akey] = A
        return self.fpd['h']

    def diagnostic(self, X, Y, m):
        print('DIAGNOSTIC', '\n', 'TRANING SET INFO')
        print('training size', str(m));
        print('X:');
        print(X);
        print('Y:');
        print(Y)
        print('FORWARD PASS INFO')
        print('Thetas');
        [print(key, value.shape) for key, value in self.weights.items()]
        print('Forward pass data');
        [print(key, value.shape) for key, value in self.fpd.items()]


def sigmoid(z):
    return 1. / (1. + np.exp(-1. * z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (sigmoid(1 - z)))


def cost(h, Y, NN, lamb, m):
    J1 = np.sum(np.matmul(Y, np.log(h).T) + np.matmul((1 - Y), np.log(1 - h).T))
    reg = 0
    Thetas = [value for item, value in NN.weights.items()]
    for Theta in Thetas:
        val = np.sum(np.square(Theta))
        reg += val
    reg /= (lamb / (2 * m))
    J = (-1 * J1) / m
    return J


def plot(costs, lr, arch):
    training_times = len(costs)
    iterations = list(range(1, training_times + 1))
    fig, ax = plt.subplots()
    plt.xlabel('Iterations');
    plt.ylabel('Cost value')
    arch = ' '.join([str(layer) for layer in arch])
    plt.title('Cost VS Iterations' + ' Lr: ' + str(lr) + ' Arch: ' + arch)
    ax.scatter(iterations, costs)
    plt.show()


def Neural_Network(X, Y, m, architecture, learning_rate, lamb, iterations):
    Neural_net = NN(architecture)
    cost_data = []
    for i in range(iterations):
        print('loop', i)
        # Neural_net.diagnostic(X, Y, m)
        Neural_net.forward_pass(X, m)
        h = Neural_net.fpd['h']
        current_cost = cost(h, Y, Neural_net, lamb, m)
        cost_data.append(current_cost)
        print('########################', '\n', 'COST IS:', str(current_cost))
        # [print(key, value.shape) for key, value in Neural_net.fpd.items()]
        Neural_net.back_propagation(X, Y, m)
        Neural_net.update(learning_rate)
        sleep(0)
    ans = Neural_net.forward_input(X, m)
    print(ans)
    plot(cost_data, learning_rate, architecture)