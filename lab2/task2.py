import numpy as np

def gauss_seidel(A, b, d_error=0.001, max_iter=15):
    # кількість рівнянь
    equations_number = len(b)
    # початкові наближення х = [0. 0. 0. 0.]
    x = np.zeros(equations_number)


    print("System of Equations:")
    for i in range(equations_number):
        equation_str = " + ".join([f"{A[i, j]}x{j + 1}" for j in range(equations_number)]) + f" = {b[i]}"
        print(f"{equation_str}")

    print("\nGauss-Seidel Iterations:")
    print("Iteration\t x1\t\t x2\t\t x3\t\t x4\t\t Error")
    print("-----------------------------------------------------------")

    for k in range(max_iter):
        x_prev = x.copy()
        x[0] = (b[0] - A[0, 1] * x_prev[1] - A[0, 2] * x_prev[2] - A[0, 3] * x_prev[3]) / A[0, 0]
        x[1] = (b[1] - A[1, 0] * x_prev[0] - A[1, 2] * x_prev[2] - A[1, 3] * x_prev[3]) / A[1, 1]
        x[2] = (b[2] - A[2, 0] * x_prev[0] - A[2, 1] * x_prev[1] - A[2, 3] * x_prev[3]) / A[2, 2]
        x[3] = (b[3] - A[3, 0] * x_prev[0] - A[3, 1] * x_prev[1] - A[3, 2] * x_prev[2]) / A[3, 3]

        error = np.max(np.abs(x - x_prev))
        print(f"Iteration {k + 1}:\t\t {x[0]:.4f}\t\t {x[1]:.4f}\t\t {x[2]:.4f}\t\t {x[3]:.4f}\t\t {error:.4f}")

        if error < d_error:
            break

    print("\nSolution:")
    print(f"x1 = {x[0]:.4f}, x2 = {x[1]:.4f}, x3 = {x[2]:.4f}, x4 = {x[3]:.4f}")

# Задана система рівнянь
A = np.array([[8, 40, -3, 0],
              [-7, 5, 0, 50],
              [8, 0, 64, -11],
              [32, 0, 0, 5]])

b = np.array([28, 0, 18, 12])

# Виклик функції для знаходження розв'язку
gauss_seidel(A, b)
