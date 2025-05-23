import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=784,
                                     hidden_size_list=[100, 100, 100, 100, 100],
                                     output_size=10,
                                     weight_init_std=weight_init_std,
                                     use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784,
                                  hidden_size_list=[100, 100, 100, 100, 100],
                                  output_size=10,
                                  weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(int(max_epochs * iter_per_epoch)):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(i / iter_per_epoch), " | " + str(train_acc) + " - " + str(bn_train_acc))

    return train_acc_list, bn_train_acc_list
if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    x_train = x_train[:1000]
    t_train = t_train[:1000]

    max_epochs = 20
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    weight_scale_list = np.logspace(0, -4, num=16)
    x = np.arange(max_epochs)

    for i, w in enumerate(weight_scale_list):
        print("============== " + str(i+1) + "/16" + " ==============")
        train_acc_list, bn_train_acc_list = __train(w)

        plt.subplot(4, 4, i + 1)
        plt.title("W:" + str(w))
        if i == 15:
            plt.plot(x, bn_train_acc_list,
                    label='Batch Normalization', markevery=2)
            plt.plot(x, train_acc_list, linestyle='--', label='Normal(without BN)', markevery=2)

        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, markevery=2)
        
        plt.ylim(0, 1.0)
        if i % 4:
            plt.yticks([])
        else:
            plt.ylabel("accuracy")
        if i < 12:
            plt.xticks([])
        else:
            plt.xlabel("epochs")
        plt.legend(loc='lower right')

    plt.show()
