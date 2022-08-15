from clean import clean
from missing import find_missing
from results_processing import find_best, aggregate_results
from aug_experiments import run_best
from calc_stat import statistical_tests
from compare import aug, improvement


if __name__ == '__main__':
    clean()
    find_missing()
    find_best()
    run_best()
    statistical_tests()
    aug()
    improvement()
    aggregate_results()
