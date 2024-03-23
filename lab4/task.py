import numpy as np

def gausa(A, b):
    n = len(b)
    Ab = np.column_stack((A, b))

    for i in range(n):
        max_row_index = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row_index]] = Ab[[max_row_index, i]]

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

def transformLamdaTpSystem(Lambda):
    system = []

    for i in range(len(Lambda)):
        line = []
        for j in range(len(Lambda[i])):
            if j == len(Lambda[i]) - 1:
                line.append(1)
            elif i == j:
                a = 0 - np.sum(Lambda[i])
                line.append(a)
            else:
                line.append(Lambda[i][j])
        
        system.append(line)
    
    new_system = np.array(system)
    return new_system.T

# Задана матриця інтенсивності переходів
Lambda = np.array([[0,   2, 1, 0   ],
                   [1.5, 0, 0, 0.02],
                   [1.2, 0, 0, 0.03],
                   [0.5, 0, 0, 0   ]])

# Задана система рівнянь
system = transformLamdaTpSystem(Lambda)
answers = np.array([0, 0, 0, 1,])

solution = gausa(system, answers)
for i in range(len(solution)):
    print(f"p[{i+1}]: {solution[i]:^10.3f}")