import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# 粒子群算法参数配置
class args:
    W = 0.5 # 惯性权重
    c1 = 0.2 # 局部学习因子
    c2 = 0.5 # 全局学习因子
    n_iterations = 10 # 迭代步数
    n_particles = 100 # 粒子数

# SVM配置
kernel = 'rbf' # ["linear","poly","rbf","sigmoid"]


def fitness_function(position,data):
    X_train, X_test, y_train, y_test = data
    svclassifier = SVC(kernel=kernel, gamma = position[0], C = position[1] )
    svclassifier.fit(X_train, y_train)
    y_train_pred = svclassifier.predict(X_train)
    y_test_pred = svclassifier.predict(X_test)
    return confusion_matrix(y_train,y_train_pred)[0][1] + confusion_matrix(y_train,y_train_pred)[1][0], confusion_matrix(y_test,y_test_pred)[0][1] + confusion_matrix(y_test,y_test_pred)[1][0]


def pso_svm(data):
    # 初始化参数
    particle_position_vector = np.array([np.array([random.random() * 10, random.random() * 10]) for _ in range(args.n_particles)])
    pbest_position = particle_position_vector
    pbest_fitness_value = np.array([float('inf') for _ in range(args.n_particles)])
    gbest_fitness_value = np.array([float('inf'), float('inf')])
    gbest_position = np.array([float('inf'), float('inf')])
    velocity_vector = ([np.array([0, 0]) for _ in range(args.n_particles)])
    iteration = 0

    while iteration < args.n_iterations:
        # plot(particle_position_vector)
        for i in range(args.n_particles):
            fitness_cadidate = fitness_function(particle_position_vector[i], data)
            print("error of particle-", i, "is (training, test)", fitness_cadidate, " At (gamma, c): ",
                  particle_position_vector[i])

            if (pbest_fitness_value[i] > fitness_cadidate[1]):
                pbest_fitness_value[i] = fitness_cadidate[1]
                pbest_position[i] = particle_position_vector[i]

            if (gbest_fitness_value[1] > fitness_cadidate[1]):
                gbest_fitness_value = fitness_cadidate
                gbest_position = particle_position_vector[i]
            elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] > fitness_cadidate[0]):
                gbest_fitness_value = fitness_cadidate
                gbest_position = particle_position_vector[i]

        for i in range(args.n_particles):
            new_velocity = (args.W * velocity_vector[i]) + (args.c1 * random.random()) * (
                        pbest_position[i] - particle_position_vector[i]) + (args.c2 * random.random()) * (
                                       gbest_position - particle_position_vector[i])
            new_position = new_velocity + particle_position_vector[i]
            particle_position_vector[i] = new_position

        iteration = iteration + 1