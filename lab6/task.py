import numpy as np
from prettytable import PrettyTable


def function(x):
    return 2*x**3 - 5 *x**2 - 60*x + 200


def halving_interval_method(bounds, eps):
    table = PrettyTable()
    table.field_names = ["k", "a", "b", "x", "|b-a|"]
    k=0 

    while True:
        x = np.mean(bounds)
        l = np.absolute(bounds[1] - bounds[0]) 
        y = function(x)

        formatted_items = [format(x, '.4f') for x in [bounds[0], bounds[1], x, l]]
        table.add_row([k] + formatted_items)
        k+=1

        if l < eps:
            print(table)
            return x, y
            
        if function(bounds[0] + l / 4) < function(bounds[0] + 3 * l / 4):
            bounds = [bounds[0], bounds[0] + l / 2]
        else:
            bounds = [bounds[0] + l / 2, bounds[1]]

    
def golden_ratio_method(bounds, eps):
    phi = 0.5 * (1 + np.sqrt(5))
    a, b = bounds
    table = PrettyTable()
    table.field_names = ["k", "a", "b", "x", "|b-a|"]
    k=0

    while True:
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi

        f1 = function(x1)
        f2 = function(x2)

        if f1 < f2:
            b = x2  
        else:
            a = x1

        formatted_items = [format(x, '.4f') for x in [a, b, (a + b) / 2, (b-a) / 2]]
        table.add_row([k] + formatted_items)
        k+=1

        if abs(b - a) < eps:
            print(table)
            return (a + b) / 2, function((a + b) / 2)


def main():
    eps = 0.01
    bounds = [0, 10]

    print("Цільова функція f(x) = 2x^3 - 5x^2 - 60x + 200")
    print("[a, b] = [", bounds[0], ',', bounds[1], ']')
    print("eps = ", eps)
    print("\nМетод половинного ділення інтервалу:")

    halving_interval_x, halving_interval_y = halving_interval_method(bounds, eps)
    print("Оптимальне значення х = ", halving_interval_x)
    print("Мінімальне значення функції f(x) = ", halving_interval_y)

    print("\nМетод золотого перерізу:")
    golden_ratio_x, golden_ratio_y = golden_ratio_method(bounds, eps)
    print("Оптимальне значення х = ", golden_ratio_x)
    print("Мінімальне значення функції f(x) = ", golden_ratio_y)


main()