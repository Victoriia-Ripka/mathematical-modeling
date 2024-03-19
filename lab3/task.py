import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, y0, x0, xn, h):
    n = int((xn - x0) / h)
    x_values = np.linspace(x0, xn, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        y_values[i] = y_values[i-1] + h * f(x_values[i-1], y_values[i-1])

    return x_values, y_values

def euler_cauchy_method(f, y0, x0, xn, h):
    n = int((xn - x0) / h)
    x_values = np.linspace(x0, xn, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        y_half = y_values[i-1] + 0.5 * h * f(x_values[i-1], y_values[i-1])
        y_values[i] = y_values[i-1] + h * f(x_values[i-1] + 0.5 * h, y_half)

    return x_values, y_values

def improved_euler_method(f, y0, x0, xn, h):
    n = int((xn - x0) / h)
    x_values = np.linspace(x0, xn, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        y_predictor = y_values[i-1] + h * f(x_values[i-1], y_values[i-1])
        y_values[i] = y_values[i-1] + 0.5 * h * (f(x_values[i-1], y_values[i-1]) + f(x_values[i], y_predictor))

    return x_values, y_values

def runge_kutta_4(f, y0, x0, xn, h):
    n = int((xn - x0) / h)
    x_values = np.linspace(x0, xn, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0

    for i in range(1, n+1):
        k1 = h * f(x_values[i-1], y_values[i-1])
        k2 = h * f(x_values[i-1] + 0.5 * h, y_values[i-1] + 0.5 * k1)
        k3 = h * f(x_values[i-1] + 0.5 * h, y_values[i-1] + 0.5 * k2)
        k4 = h * f(x_values[i-1] + h, y_values[i-1] + k3)
        y_values[i] = y_values[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x_values, y_values

def f(x, y):
    return y + np.sin(x/2.8)

# Задані параметри
x0, xn = 1.4, 2.3999
y0 = 2.2
h = 0.1

# Виклик методів
x_euler, y_euler = euler_method(f, y0, x0, xn, h)
x_euler_cauchy, y_euler_cauchy = euler_cauchy_method(f, y0, x0, xn, h)
x_improved_euler, y_improved_euler = improved_euler_method(f, y0, x0, xn, h)
x_rk4, y_rk4 = runge_kutta_4(f, y0, x0, xn, h)

# Точний розв'язок для порівняння
x_exact = np.linspace(x0, xn, 10)
y_exact = 2.2 * np.exp(x_exact - 1.4) - np.sin(x_exact / 2.8) + np.sin(1.4 / 2.8)
print(x_exact, '\n', y_exact, '\n\n')


if len(y_exact) == len(y_euler):
    print('euler:          ', y_euler)
    print('error delta:    ', abs(y_exact-y_euler))
else:
    print('Розмірності не співпадають. Неможливо порівняти.')

print('euler_cauchy:   ', y_euler_cauchy)
print('error delta:    ', abs(y_exact-y_euler_cauchy))
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
