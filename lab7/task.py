import numpy as np
from prettytable import PrettyTable


def function(x):
    return (x[0]**2 + x[1]**2 - 1)**2 + (x[0] + x[1] - 1)**2


def gradient(x):
    df_dx0 = 2 * (x[0]**2 + x[1]**2 - 1) * 2 * x[0] + 2 * (x[0] + x[1] - 1)
    df_dx1 = 2 * (x[0]**2 + x[1]**2 - 1) * 2 * x[1] + 2 * (x[0] + x[1] - 1)
    return np.array([df_dx0, df_dx1])


def format_float_list(lst):
    return '[' + ', '.join([f'{x:.4f}' for x in lst]) + ']'


def gradient_optimize_method(x, eps1, eps2, M, a):
    k = 0
    table = PrettyTable()
    table.field_names = ["iter", "x", "||gradient||", "a", "x_next", "||x_next - x||", "|f(x_next) - f(x)|"]    

    while True:
        x = gradient(x)
        print(x)
        termination_criterion = np.sqrt(np.sum(np.square(x)))
        print(termination_criterion)

        if termination_criterion < eps1 or k == M:
            break
        
        x_next = [x[0] - a*df_dx0, x[1] - a*df_dx1]

        row = [format_float_list(x), "{:.4f}".format(termination_criterion), "{:.4f}".format(a), format_float_list(x_next), "{:.4f}".format(np.linalg.norm(np.array(x_next) - np.array(x))), "{:.4f}".format(np.abs(function(x_next) - function(x)))]
        table.add_row([k+1] + row)

        if function(x_next) - function(x) < 0:
            break
        else:
            a = a/2

        if np.abs(function(x_next) - function(x)) < eps2:
            break 
        
        k += 1
        x = x_next
    

    # print(table)
    return [df_dx0, df_dx1]

def main():
    eps1, eps2 = 0.1, 0.15
    M = 10
    a = 0.5
    x = [0, 3]

    print("Цільова функція: y = (x0^2 + x1^2 - 1)^2 + (x0 + x1 - 1)^2")
    print("Початкова точка: ", x)
    print("a =", a, ", eps1 =", eps1, ", eps2 =", eps2, ", M =", M)
    print("Метод градієнтного спуску із постійним кроком")

    answer = gradient_optimize_method(x, eps1, eps2, M, a)
    print(answer)

main()
