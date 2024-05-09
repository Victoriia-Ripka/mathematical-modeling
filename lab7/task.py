import numpy as np
from prettytable import PrettyTable


def function(x):
    return (x[0]**2 + x[1]**2 - 1)**2 + (x[0] + x[1] - 1)**2


def gradient(x):
    df_dx0 = 4 * x[0] * (x[0]**2 + x[1]**2 - 1) + 2 * (x[0] + x[1] - 1)
    df_dx1 = 4 * x[1] * (x[0]**2 + x[1]**2 - 1) + 2 * (x[0] + x[1] - 1)
    return np.array([df_dx0, df_dx1])


def format_float_list(lst):
    return '[' + ', '.join([f'{x:.4f}' for x in lst]) + ']'


def gradient_optimize_method(x, eps1, eps2, M, a):
    # step 2
    k = 0
    k_dict = {}

    table = PrettyTable()
    table.field_names = ["iter", "x", "||gradient||", "a", "x_next", "||x_next - x||", "|f(x_next) - f(x)|"]    

    while True:
        k_dict[k] = False

        # step 3
        grad = gradient(x)
        grad_abs = np.sqrt(np.sum(np.square(grad)))

        # step 4 and 5
        if grad_abs < eps1 or k == M:
            print("break termination_criterion < eps1 or k == M")
            print(table)
            return x
        
        # step 6
        if k == 0:
            a_step = a
        else:
            a_step = a / 2

        # step 7
        x_next = [x[0] - a_step*grad[0], x[1] - a_step*grad[1]]

        row = [format_float_list(x), "{:.4f}".format(grad_abs), "{:.4f}".format(a_step), format_float_list(x_next), "{:.4f}".format(np.linalg.norm(np.array(x_next) - np.array(x))), "{:.4f}".format(np.abs(function(x_next) - function(x)))]
        table.add_row([k+1] + row)

        # step 8
        if function(x_next) - function(x) < 0:
            # крок 9
            func_abs = np.abs(function(x_next) - function(x))
            x_distance = np.linalg.norm(np.array(x_next) - np.array(x))
            if func_abs < eps2 and x_distance < eps2:
                k_dict[k] = True

                # step 9a
                if k_dict[k-1] == True:
                    print("break np.abs(function(x_next) - function(x)) < eps2")
                    print(table)
                    return x_next
            else:
                x = x_next.copy()
                k += 1
                continue
        else:
            while function(x_next) - function(x) > 0:
                a_step = a_step/2
                x_next = np.array([x[0] - a_step*grad[0], x[1] - a_step*grad[1]]).copy()
            

        # крок 9
        func_abs = np.abs(function(x_next) - function(x))
        x_distance = np.linalg.norm(np.array(x_next) - np.array(x))
        if func_abs < eps2 and x_distance < eps2:
            k_dict[k] = True

            # step 9a
            if k_dict[k-1] == True:
                print(table)
                return x_next
            
        k += 1
        x = x_next.copy()


def main():
    eps1, eps2 = 0.05, 0.15
    M = 10
    a = 0.5
    x = [0, 3]

    print("Цільова функція: y = (x0^2 + x1^2 - 1)^2 + (x0 + x1 - 1)^2")
    print("Початкова точка: ", x)
    print("a =", a, ", eps1 =", eps1, ", eps2 =", eps2, ", M =", M)
    print("Метод градієнтного спуску із постійним кроком\n\n")

    x_answer = gradient_optimize_method(x, eps1, eps2, M, a)
    print("Оптимальне значення: (", x_answer[0], ",", x_answer[1], ")")
    func = function(x_answer)
    print("Мінімальне значення функції: ", func)

main()
