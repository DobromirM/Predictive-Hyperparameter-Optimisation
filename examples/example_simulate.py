import pandas as pd

from examples.helper import define_hp_searcher_simulation

from examples.helper import calculate_random_search

if __name__ == '__main__':
    hp_searcher = define_hp_searcher_simulation()
    data = pd.read_csv('simulate.txt', header=0)

    average_accuracy = hp_searcher.simulate(data)

    print('\n')
    print(f'Maximum possible accuracy: {data.max()[1] * 100}%')
    print(f'Average accuracy for Random Search: {calculate_random_search(data.iloc[:, 1])}%')
    print(f'Average accuracy for Predictive Hyper Opt: {average_accuracy}%')
