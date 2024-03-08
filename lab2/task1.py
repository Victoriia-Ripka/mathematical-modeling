import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_values, y_values, x):
    result = 0
    formula = ""
    for i in range(len(y_values)):
        term = y_values[i]
        term_str = str(y_values[i])
        for j in range(len(x_values)):
            if j != i:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
                term_str += f" * (x - {x_values[j]}) / ({x_values[i]} - {x_values[j]})"
        result += term
        if i > 0:
            formula += " + " + term_str
        else:
            formula += term_str

    return result, formula

# Задані точки
x_values = [-2, 0, 1, 2]
y_values = [-11, -3, -11, 5]

# Задані точки для обчислення L(x1), L(x2), L(x3), L(x4)
specified_points = [-1.5, -0.5, 0.5, 2.5]

# Обчислюємо значення інтерполяційного многочлена та формулу для заданих точок
result_values = []
formulas = []
for x in specified_points:
    result, formula = lagrange_interpolation(x_values, y_values, x)
    result_values.append(result)
    formulas.append(formula)

# Вивід результатів обчислень
for i in range(len(specified_points)):
    print(f"L({specified_points[i]}) = {result_values[i]} (Formula: {formulas[i]})")

# Побудова графіку інтерполяційної функції
x_plot = np.linspace(min(x_values), max(x_values), 1000)
y_plot = [lagrange_interpolation(x_values, y_values, x)[0] for x in x_plot]

plt.plot(x_plot, y_plot, label='Lagrange Interpolation')
plt.scatter(x_values, y_values, color='red', label='Given Points')
plt.scatter(specified_points, result_values, color='blue', label='Specified Points')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Lagrange Interpolation and Given Points')
plt.grid(True)
plt.show()