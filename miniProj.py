import numpy as np
import math
import matplotlib.pyplot as plt


# Projection of u on to X is same as the projection of w onto C
# An Euclidean projection of a point x_0 on a set S is a point that min ||x - x_0|| (x belongs to S)
# In scenario 1, if the x_ or w_ is out of hypercube, we will reduce the dimension, which is beyond [-1, 1].
def projection_1(w):
    w = np.where( w >= 1, 1, w)
    w = np.where(w <= -1, -1, w)
    return w

# In scenario 2, if x_ or w_ is out of the unit ball, we need to normalized the vector.
def projection_2(w):
    norm = np.sum(np.square(w))
    if norm > 1:
        w /= norm
    return w


# generate training and test examples for each scenario

# numpy.random.multivariable_normal(mean, cov, [.size])
# input:  mean "mu", covariance matrix "sigma" and the size of the samples.
# output: data draw from the distribution D
# We use this function to draw random samples from a multivariate normal distribution.

def generate_data(n, sigma, scenario):
    py = np.random.rand(n, 1)
    y = np.where(py >= 0.5, 1, -1)
    x = np.zeros((n, 4))
    mu_0 = np.ones(4) * -0.25
    mu_1 = np.ones(4) * 0.25
    idmtrx = np.identity(4)


    idx0 = np.where(py < 0.5)[0]
    idx1 = np.where(py >= 0.5)[0]
    x[idx0, :] = np.random.multivariate_normal(mu_0, sigma**2 * idmtrx, len(idx0))
    x[idx1, :] = np.random.multivariate_normal(mu_1, sigma**2 * idmtrx, len(idx1))
    # plt.scatter(x[:,0], x[:, 1])
    # plt.show()

    if scenario == 1:
        x = projection_1(x)
    else:
        for j in range(x.shape[0]):
            x[j, :] = projection_2(x[j, :])

    return y, x


# M: parameter set C is M-bounded
# rho: the loss function id rho-Lipschitz
# rate: the step size of the SGD
# rate = M/(rho * sqrt(t))
# Given w_t, we use G-Oracle to generates a random vector Gt
def SGD(w, x, y, n, scenario):

    one = np.ones((n, 1))
    x_ = np.hstack((x, one))
    w_ = np.zeros((n, 5))

    if scenario == 1:
        M = math.sqrt(5)
        rho = math.sqrt(5)
    else:
        M = 1
        rho = math.sqrt(2)

    #print rate
    for t in range(1, n):
        rate_ = M / rho /math.sqrt(t)
        # G-Oracle(wt)
        G = np.divide(-y[t] * x_[t], 1 + math.exp(y[t] * np.sum(w * x_[t])))
        w = w - rate_ * G
        if scenario ==1:
            w = projection_1(w)
        else:
            w = projection_2(w)
        w_[t] = w

    return np.mean(w_, axis=0)

# the binary classification error function
def err(w, x, y, N):
    one = np.ones((N,1))
    x_ = np.hstack((x, one))
    res = np.dot(x_, w.reshape(5, 1))
    y_ = np.where(res >= 0, 1, -1)
    error = float(np.sum(y != y_)) / N
    return error

def calc(scenario, n, sigma, x_test, y_test):

    y_train, x_train = generate_data(n, sigma, scenario)

    w = np.zeros(5)

    w = SGD(w, x_train, y_train, n, scenario)
    error = err(w, x_test, y_test, N)

    if error < 0:
        print error

    return error

# plot the estimate of the expected classification error of the SGD learner
if __name__ == '__main__':
    scenario = [1, 2]
    N = 400
    n = [50, 100, 500, 1000]
    sigma = [0.05, 0.25]

    lines = []
    legends = []
    for i in range(2):
        for j in range(2):
            y_test, x_test = generate_data(N, sigma[j], scenario[i])
            error = np.zeros((4, 20))
            for k in range(4):
                for m in range(20):
                    error[k][m] = calc(scenario[i], n[k], sigma[j], x_test, y_test)
                    if error[k][m] < 0:
                        print error[k][m]
            std = np.std(error, axis=1)
            mu = np.mean(error, axis=1)
            print std, mu
            # plt.plot(n, np.mean(error, axis=1))
            line = plt.errorbar(n, mu, yerr=std)
            lines.append(line)
            legends.append('scenario %d simga %.2f' % (scenario[i], sigma[j]))
    plt.legend(lines, legends)
    plt.show()




