import numpy as np
from prettytable import PrettyTable


def function(x):
    return 2*x[0]**2 +x[0]*x[1] + x[1]**2


def gradient(x):
    df_dx0 = 4*x[0] + x[1]
    df_dx1 = x[0] + 2*x[1]
    return np.array([df_dx0, df_dx1])


def format_float_list(lst):
    return '[' + ', '.join([f'{x:.4f}' for x in lst]) + ']'


def gradient_optimize_method(x, eps1, eps2, M, a):
    # step 2
    k = 0
    table = PrettyTable()
    table.field_names = ["iter", "x", "||gradient||", "a", "x_next", "||x_next - x||", "|f(x_next) - f(x)|"]    

    while True:
        # step 3
        grad = gradient(x)
        termination_criterion = np.sqrt(np.sum(np.square(grad)))

        # step 4 and 5
        if termination_criterion < eps1 or k == M:
            print("break termination_criterion < eps1 or k == M")
            return x
        
        # step 6
        a_step = a / (k + 1)

        # step 7
        x_next = [x[0] - a*grad[0], x[1] - a*grad[1]]

        row = [format_float_list(x), "{:.4f}".format(termination_criterion), "{:.4f}".format(a), format_float_list(x_next), "{:.4f}".format(np.linalg.norm(np.array(x_next) - np.array(x))), "{:.4f}".format(np.abs(function(x_next) - function(x)))]
        table.add_row([k+1] + row)

        # step 8
        if function(x_next) - function(x) < 0:
            # крок 9
            if np.all(np.abs(function(x_next) - function(x)) < eps2) and np.all(np.sqrt(np.subtract(np.square(x_next), np.square(x))) < eps2) :
                print("break np.abs(function(x_next) - function(x)) < eps2")
                return x_next
            else:
                x = x_next.copy()
                k += 1
                continue
        else:
            while function(x_next) - function(x) > 0:
                a_step = a_step/2
                x_next = np.array([x[0] - a_step*grad[0], x[1] - a_step*grad[1]]).copy()
                print(x, x_next)
            

        # крок 9
        if np.all(np.abs(function(x_next) - function(x)) < eps2) and np.all(np.sqrt(np.subtract(np.square(x_next), np.square(x))) < eps2) :
            return x_next  # a or b step9
        
        k += 1
        x = x_next.copy()


def main():
    eps1, eps2 = 0.1, 0.15
    M = 10
    a = 0.5
    x = [0.5, 1]

    print("Цільова функція: y = 2x1^2 +x1*x2 + x2^2")
    print("Початкова точка: ", x)
    print("a =", a, ", eps1 =", eps1, ", eps2 =", eps2, ", M =", M)
    print("Метод градієнтного спуску із постійним кроком")

    answer = gradient_optimize_method(x, eps1, eps2, M, a)
    print(answer)

main()
