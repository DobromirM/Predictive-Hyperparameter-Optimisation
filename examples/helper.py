from algorithm.predictive_hyper_opt_algorithm import PredictiveHyperOpt
from scipy.special import comb


def define_hp_searcher_simulation():
    return PredictiveHyperOpt(total_models_count=200, full_models_count=10, final_epoch=45, partial_epochs=[4, 1], batch_size=1024,
                              final_models_count=20)


def define_hp_searcher_normal():
    return PredictiveHyperOpt(total_models_count=100, full_models_count=5, final_epoch=10, partial_epochs=[2, 1], final_models_count=10,
                              batch_size=250)


def calculate_random_search(accuracies, sample_size=45):
    accuracies = accuracies.sort_values().reset_index(drop=True)

    length = len(accuracies)

    total_choices = 0
    cumulative_value = 0

    for i in range(0, length - sample_size + 1):
        choices = comb(sample_size - 1 + i, sample_size - 1)
        total_choices = total_choices + choices
        cumulative_value = cumulative_value + accuracies[i + sample_size - 1] * choices

    return round(cumulative_value / total_choices * 100, ndigits=2)


def perform_random_search(generator, x_train, y_train, x_test, y_test, count=32, epochs=10, batch_size=250):
    final_accuracies = list()

    for i in range(0, count):
        print(f'\nRandom search: {i}')
        model = generator.create_mutated_model()
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        validation_stats = model.evaluate(x_test, y_test)

        final_accuracies.append(validation_stats[1])

    return round(max(final_accuracies) * 100, ndigits=2)
