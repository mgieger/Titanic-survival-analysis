from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from dtree.decisiontree import DecisionTreeRunner
import pprint


def main():
    """ columns = [
        'pclass',
        'name',
        'sex',
        'age',
        'sibSp',
        'parch',
        'ticket',
        'fare',
        'cabin',
        'embarked',
        'survived',
        'boat',
        'body'
        ] """
    full_file = '../titanic_full.csv'

    default_rfc = RandomForestClassifier(random_state=0)
    warmstart_rfc = RandomForestClassifier(warm_start=True)
    random_rfc = RandomForestClassifier(random_state=1)
    est_rfc = RandomForestClassifier(n_estimators=10, random_state=0)
    depth_rfc = RandomForestClassifier(max_depth=3, random_state=0)
    gradient_boost = GradientBoostingClassifier(n_estimators=1, learning_rate=.1)
    combo_1_rfc = RandomForestClassifier(n_estimators=9, random_state=0, max_depth=3)
    extreme_rand_rfc = RandomForestClassifier(max_depth=3, min_samples_split=10, random_state=0, n_estimators=9)


    preprocessor = Preprocessor(full_file)
    data_perm_1 = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare', 'pclass', 'name', 'sibsp', 'parch',  'embarked', 'survived']) #2
    data_perm_5 = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare', 'survived'])  #2
    data_perm_4 = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare','cabin', 'survived'])
    data_perm_10 = preprocessor.get_matrix_split(['sex', 'age', 'fare', 'name', 'survived']) #suprisingly the worst despite
    data_perm_9 = preprocessor.get_matrix_split(['sex', 'age', 'survived']) # 1) best b4 tuning - but two parameters seems useless
    #being made up of the features that were deemed to have the highest immportance

    experiment_result_dicts = []

    # just randomly play around with permutations of feature sets
    print("perm 1 default")
    results_tree_1 = run_experiment(data_perm_1, default_rfc)
    print("\ntree perm 9")
    results_tree_9 = run_experiment(data_perm_9, default_rfc)

    print("\ntree perm 10")
    results_tree_10 = run_experiment(data_perm_10, default_rfc)

    print("\nperm 1 warm")
    results_tree_1_warm = run_experiment(data_perm_1, warmstart_rfc)

    print("\nperm 1 gradient boost forest")
    gradient_boost_tree = run_experiment(data_perm_1, gradient_boost)

    print("\nperm 1 random")
    results_tree_1_random = run_experiment(data_perm_1, random_rfc)

    print("\nperm 1 est")
    results_tree_1_perm = run_experiment(data_perm_1, est_rfc)

    print("\nperm 1 depth")
    results_tree_1_depth = run_experiment(data_perm_1, depth_rfc)

    print("\nperm 1 combo 1")
    results_tree_1_combo_1 = run_experiment(data_perm_1, combo_1_rfc)
    print("\ncombo exp perm 1")

    print("\nperm 5 combo 1")
    results_tree_5_combo_1 = run_experiment(data_perm_5, combo_1_rfc)

    print("\nperm 9 combo 1")
    results_tree_9_combo_1 = run_experiment(data_perm_9, combo_1_rfc)

    print("\nperm 10 combo 1")
    results_tree_10_combo_1 = run_experiment(data_perm_10, combo_1_rfc)

    #TODO: use perm 4 to try to optimize for best results
    print("\nperm 4 combo 1")
    results_tree_4_combo_1 = run_experiment(data_perm_4, combo_1_rfc)

    print("\n extreme random")
    results_tree_1_ext = run_experiment(data_perm_1, extreme_rand_rfc)


    print("\n ")
    pprint.pprint(experiment_result_dicts)


def run_experiment(data_perm, rfc):
    """

    :param data_perm:
    :param rfc:
    :return: dict containing experiment results
    """
    dtr = DecisionTreeRunner(data_perm, rfc)
    dtr.run()
    dtr.print_feature_importance()
    dtr.print_accuracy()
    return dtr.prediction_results
# compared failed predictions, are there any outliers who are consistent across all models?

if __name__ == '__main__':
    main()
