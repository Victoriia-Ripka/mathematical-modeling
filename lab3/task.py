import numpy as np
import matplotlib.pyplot as plt

def euler_method(x0, xn, y0, h):
    n = int((xn - x0) / h)
    x_values = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        y_values[i] = y_values[i-1] + h * f(x_values[i-1], y_values[i-1])

    return x_values, y_values

def euler_cauchy_method(x0, xn, y0, h):
    n = int((xn - x0) / h)
    x_values = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
    y_values = np.zeros(n+1)
    y_values[0] = y0

    y_values[1] = y_values[0] + h * f(x_values[0], y_values[0])

    for i in range(2, n+1):
        spec_y = y_values[i-1] + h * f(x_values[i-1], y_values[i-1])
        y_values[i] = y_values[i-1] + 0.5 * h * ( f(x_values[i-1], y_values[i-1]) + f(x_values[i], spec_y))

    return x_values, y_values

def improved_euler_method(x0, xn, y0, h):
    n = int((xn - x0) / h)
    x_values = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        spec_y = y_values[i-1] + 0.5 * h * f(x_values[i-1], y_values[i-1])
        y_values[i] = y_values[i-1] + h * f(x_values[i-1] + h * 0.5, spec_y)

    return x_values, y_values

def runge_kutta_4(x0, xn, y0, h):
    n = int((xn - x0) / h)
    x_values = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        k0 = f(x_values[i-1], y_values[i-1])
        k1 = f(x_values[i-1] + 0.5 * h, y_values[i-1] + 0.5 * h * k0)
        k2 = f(x_values[i-1] + 0.5 * h, y_values[i-1] + 0.5 * h * k1)
        k3 = f(x_values[i-1] + h, y_values[i-1] + h * k2)
        y_values[i] = y_values[i-1] + (k0 + 2*k1 + 2*k2 + k3) * h / 6

    return x_values, y_values

def f(x, y):
    return y + np.sin(x/2.8)

# Задані параметри
x0, xn = 1.4, 2.4
y0 = 2.2
h = 0.1

# Виклик методів
x_euler, y_euler = euler_method(x0, xn, y0, h)
x_euler_cauchy, y_euler_cauchy = euler_cauchy_method(x0, xn, y0, h)
x_improved_euler, y_improved_euler = improved_euler_method(x0, xn, y0, h)
x_rk4, y_rk4 = runge_kutta_4(x0, xn, y0, h)

# Точний розв'язок для порівняння
x_exact = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4])
y_exact = np.array([2.2, 2.482407, 2.799851, 3.152735, 3.545817, 3.983248, 4.469607, 5.009956, 5.609882, 6.275559, 7.013805])
print(x_exact, '\n', y_exact, '\n\n')


if len(y_exact) == len(y_euler):
    print('euler:          ', y_euler)
    print('error delta:    ', abs(y_exact - y_euler))
else:
    print('Розмірності не співпадають. Неможливо порівняти.')

print('euler_cauchy:   ', y_euler_cauchy)
print('error delta:    ', abs(y_exact - y_euler_cauchy))
print('improved_euler: ', y_improved_euler)
print('error delta:    ', abs(y_exact-y_improved_euler))
print('runge_kutta:    ', y_rk4)
print('error delta:    ', abs(y_exact-y_rk4))

# Графіки
plt.plot(x_euler, y_euler, label='Euler')
plt.plot(x_euler_cauchy, y_euler_cauchy, label='Euler-Cauchy')
plt.plot(x_improved_euler, y_improved_euler, label='Improved Euler')
plt.plot(x_rk4, y_rk4, label='Runge-Kutta 4')
plt.plot(x_exact, y_exact, label='Exact', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solutions of the Differential Equation')
plt.legend()
plt.grid(True)
plt.show()
