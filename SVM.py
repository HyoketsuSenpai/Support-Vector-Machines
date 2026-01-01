'''
SUPPORT VECTOR MACHINE
CoSC 4011: INTRODUCTION TO AI
BY
NAME OF THE STUDENTS		IDNO
1.	Yosef Solomon 				UGR/7358/15
2.	Tewobsta Seyoum            	UGR/3422/15
3.	Tsion Teklay 				UGR/1801/15
4.	Yosefe Tilahun 				UGR/9673/15
PROJECT INSTRUCTOR: MR.SAMUEL G.
January, 2026
'''

import numpy as np

class svm:
    def __init__(self, X, y, C=1, b=0, tol=1e-5, epsilon = 1e-8, max_iter=300):
        self.X = X
        self.y = y
        self.b = b
        self.tol = tol
        self.epsilon = epsilon
        self.C = C
        self.max_iter = max_iter
        self.m, self.n = np.shape(X)
        self.alphas = np.zeros(self.m)
        self.w = np.zeros(self.n)

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def predict(self, x):
        return (self.alphas * self.y) @ self.linear_kernel(self.X, x) + self.b

    def calculate_error(self, i):
        return self.predict(self.X[i, :]) - self.y[i]

    def solve(self, i_1, i_2):
        if i_1 == i_2:
            return 0

        a_1 = self.alphas[i_1]
        a_2 = self.alphas[i_2]

        y_1 = self.y[i_1]
        y_2 = self.y[i_2]

        b = self.b

        if y_1 != y_2:
            L = max(0, a_2 - a_1)
            H = min(self.C, self.C + a_2 - a_1)
        else:
            L = max(0, a_2 + a_1 - self.C)
            H = min(self.C, a_2 + a_1)

        if L==H:
            return 0

        x1 = self.X[i_1, :]
        x2 = self.X[i_2, :]

        k11 = self.linear_kernel(x1, x1)
        k22 = self.linear_kernel(x2, x2)
        k12 = self.linear_kernel(x1, x2)

        eta = k11 + k22 - 2 * k12

        E_1 = self.calculate_error(i_1)
        E_2 = self.calculate_error(i_2)

        s = y_1 * y_2

        if eta > 0:
            a_2_new = a_2 + y_2 * (E_1 - E_2) / eta

            if a_2_new >= H:
                a_2_new = H

            if a_2_new <= L:
                a_2_new = L
        else:
            f_1 = y_1 * (E_1 + self.b) - a_1 * k11 - s * a_2 * k12
            f_2 = y_2 * (E_2 + self.b) - s * a_1 * k12 - a_2 * k22

            L_1 = a_1 + s * (a_2 - L)
            H_1 = a_1 + s * (a_2 - H)

            objective_of_L = L_1 * f_1 + L * f_2 + 0.5 * (L_1 ** 2) * k11 + 0.5 * (L ** 2) * k22 + s * L * L_1 * k12
            objective_of_H = H_1 * f_1 + H * f_2 + 0.5 * (H_1 ** 2) * k11 + 0.5 * (H ** 2) * k22 + s * H * H_1 * k12

            if objective_of_L + self.tol < objective_of_H:
                a_2_new = L
            elif objective_of_L > objective_of_H + self.tol:
                a_2_new = H
            else:
                return 0

        if abs(a_2_new - a_2) < self.tol * (a_2_new + a_2 + self.tol):
            return 0

        a_1_new = a_1 + s * (a_2 - a_2_new)

        if a_1_new < self.epsilon:
            a_1_new = 0
        elif a_1_new > (self.C - self.epsilon):
            a_1_new = self.C

        b1 = b - E_1 - y_1 * (a_1_new - a_1) * k11 - y_2 * (a_2_new - a_2) * k12
        b2 = b - E_2 - y_1 * (a_1_new - a_1) * k12 - y_2 * (a_2_new - a_2) * k22

        if 0 < a_1_new < self.C:
            self.b = b1
        elif 0 < a_2_new < self.C:
            self.b = b2
        else:
            self.b = 0.5 * (b1 + b2)

        self.w = self.w + y_1 * (a_1_new - a_1) * x1 + y_2 * (a_2_new - a_2) * x2
        self.alphas[i_2] = a_2_new
        self.alphas[i_1] = a_1_new

        return 1

    def examine_example(self, i_2):
        y_2 = self.y[i_2]
        a_2 = self.alphas[i_2]
        E_2 = self.calculate_error(i_2)
        r_2 = E_2 * y_2

        support_vector = [i for i, a in enumerate(self.alphas) if 0 + self.epsilon < a < self.C - self.epsilon]

        if (r_2 < -self.tol and a_2 < self.C) or (r_2 > self.tol and a_2 > 0):

            if len(support_vector) > 1:
                i_1 = max(support_vector, key=lambda i: abs(self.calculate_error(i) - E_2))

                if self.solve(i_1, i_2):
                    return 1

            for i_1 in np.random.permutation(support_vector):
                if self.solve(i_1, i_2):
                    return 1

            for i_1 in np.random.permutation(self.m):
                if self.solve(i_1, i_2):
                    return 1

        return 0

    def fit(self):
        loop_cnt = 0

        num_changed = 0
        examine_all = 1

        while num_changed > 0 or examine_all:
            if loop_cnt > self.max_iter:
                break

            num_changed = 0

            if examine_all:
                for l in range(self.m):
                    num_changed += self.examine_example(l)

            else:
                for l in range(self.m):
                    if 0 < self.alphas[l] < self.C:
                        num_changed += self.examine_example(l)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

            loop_cnt += 1

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# cluster_std 3.0
X, y = make_blobs(n_samples=200,centers=2,cluster_std=2.0, random_state=42)

y = np.where(y == 0, -1, 1)


n_outliers = 20
rng = np.random.RandomState(42)
X_outliers = rng.uniform(low=-15, high=15, size=(n_outliers, X.shape[1]))
y_outliers = rng.choice([-1, 1], size=n_outliers)
X = np.vstack([X, X_outliers])
y = np.hstack([y, y_outliers])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.05,
    random_state=42,
    stratify=y
)


model = svm(X_train,y_train)
model.fit()

y_pred = model.predict(X_test)
y_pred = np.sign(y_pred)

accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy}')


plt.figure(figsize=(8, 6))

plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='r', edgecolors='k', label='+ve training data')
plt.scatter(X_train[y_train==-1,0], X_train[y_train==-1,1], c='b', edgecolors='k', label='-ve training data')

plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1], c='y', edgecolors='k', label='+ve testing data')
plt.scatter(X_test[y_test==-1,0], X_test[y_test==-1,1], c='c', edgecolors='k', label='-ve testing data')


x1 = np.linspace(min(X_train[:, 0]) - 1, max(X_train[:, 1]) + 1, 100)

x2_decision = -(model.w[0]/model.w[1])*x1 - model.b/model.w[1]
x2_positive = -(model.w[0]/model.w[1])*x1 - (model.b - 1)/model.w[1]
x2_negative = -(model.w[0]/model.w[1])*x1 - (model.b + 1)/model.w[1]

plt.plot(x1, x2_decision, 'k-', label='Decision boundary')
plt.plot(x1, x2_positive, 'r--', label='Margin +1')
plt.plot(x1, x2_negative, 'b--', label='Margin -1')

plt.legend()

plt.show()