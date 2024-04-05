import numpy as np
from prettytable import PrettyTable

def calculate_probability_state(matrix_transition_probabilities, started_values):
    probability_state = []
    new_state = started_values.copy()  

    for _ in range(len(started_values)):
        new_state = np.dot(matrix_transition_probabilities, new_state)
        probability_state.append(new_state)

    return probability_state


def main():
    started_values = np.array([1, 0, 0, 0, 0])
    matrix_transition_probabilities = np.array([[0.5, 0.2, 0.15, 0.1,  0.05], 
                                                [0,   0.3, 0.4,  0.2,  0.1],
                                                [0,   0,   0.45, 0.35, 0.2],
                                                [0,   0,   0,    0.5,  0.5],
                                                [0,   0,   0,    0,    1]])

    probability_state = calculate_probability_state(matrix_transition_probabilities.T, started_values)

    table = PrettyTable()
    table.field_names = ["step", "p1", "p2", "p3", "p4", "p5"]
    for i, item in enumerate(probability_state):
        formatted_items = [format(x, '.4f') for x in item]
        table.add_row([i] + formatted_items)

    print(table)


main()