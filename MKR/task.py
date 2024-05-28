import numpy as np
from prettytable import PrettyTable


def function(x):
    return 2*x**3 - 5 *x**2 - 60*x + 200


def derivative_func(x):
    return 6*x**2 - 10*x - 60


def midpoint_method(bounds, eps):
    table = PrettyTable()
    table.field_names = ["k", "a", "b", "x", "|f'(x)|"]
    k=0 

    while True:
        x = np.mean(bounds)
        df_x = derivative_func(x)
        f = function(x)

        formatted_items = [format(x, '.4f') for x in [bounds[0], bounds[1], x, df_x]]
        table.add_row([k] + formatted_items)
        k+=1

        if np.absolute(df_x) <= eps:
            print(table)
            return x, f
            
        if df_x > 0:
            bounds = [bounds[0], x]
        else:
            bounds = [x, bounds[1]]


def main():
    eps = 0.01
    bounds = [0, 10]

    print("Цільова функція f(x) = 2x^3 - 5x^2 - 60x + 200")
    print("[a, b] = [", bounds[0], ',', bounds[1], ']')
    print("eps = ", eps)
    print("\nМетод середньої точки:")

    halving_interval_x, halving_interval_y = midpoint_method(bounds, eps)
    print("Оптимальне значення х = ", halving_interval_x)
    print("Мінімальне значення функції f(x) = ", halving_interval_y)


main()