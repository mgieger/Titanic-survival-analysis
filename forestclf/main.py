from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from forestclf.ensemblerunner import EnsembleRunner
import pprint
import itertools as iter


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

    feature_list = ['sex', 'age', 'ticket', 'fare', 'pclass', 'name', 'sibsp', 'parch',  'embarked', 'survived']
    feature_list = ['sex', 'age', 'ticket', 'fare', 'pclass', 'name', 'sibsp', 'parch', 'embarked']
    survived = 'survived'

    default_rfc = RandomForestClassifier(random_state=0)
    warmstart_rfc = RandomForestClassifier(warm_start=True)
    random_rfc = RandomForestClassifier(random_state=1)
    est_rfc = RandomForestClassifier(n_estimators=10, random_state=0)
    depth_rfc = RandomForestClassifier(max_depth=3, random_state=0)
    gradient_boost = GradientBoostingClassifier(n_estimators=1, learning_rate=.1)
    combo_1_rfc = RandomForestClassifier(n_estimators=9, random_state=0, max_depth=3)
    extreme_rand_rfc = RandomForestClassifier(max_depth=3, min_samples_split=10, random_state=0, n_estimators=9)

    classifier_dict= {
        "default_rfc": default_rfc,
        #"warmstart_rfc": warmstart_rfc,
        "random_rfc": random_rfc,
        "est_rfc": est_rfc,
        "depth_rfc": depth_rfc,
        "gradient_boost": gradient_boost,
        "combo_1_rfc": combo_1_rfc,
        "extreme_rand_rfc": extreme_rand_rfc
    }

    max_dict = {
        "value": 0,
        "feature_list": "",
        "classifier": ""
    }

    preprocessor = Preprocessor(full_file)

    for k in classifier_dict:
        for i in range(len(feature_list)):
            if i < 1:
                continue
            for j in iter.combinations(feature_list, i):
                j = list(j)
                j.append(survived)
                data = preprocessor.get_matrix_split(list(j))
                pred_results, acc = run_experiment(data, classifier_dict[k])
                if acc > max_dict["value"]:
                    max_dict["value"] = acc
                    max_dict["feature_list"] = j
                    max_dict["classifier"] = k
    print("MAX:")
    print(max_dict)

    # data_perm_1 = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare', 'pclass', 'name', 'sibsp', 'parch',  'embarked', 'survived']) #2
    # data_perm_5 = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare', 'survived'])  #2
    # data_perm_4 = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare','cabin', 'survived'])
    # data_perm_10 = preprocessor.get_matrix_split(['sex', 'age', 'fare', 'name', 'survived']) #suprisingly the worst despite
    #data_perm_9 = preprocessor.get_matrix_split(['sex', 'age', 'survived']) # 1) best b4 tuning - but two parameters seems useless
    # #being made up of the features that were deemed to have the highest immportance
    #
    # experiment_result_dicts = []
    #
    #
    #
    # just randomly play around with permutations of feature sets
    # print("perm 1 default")
    # results_rf_1 = run_experiment(data_perm_1, default_rfc)
    #print("\nrf perm 9")
    #results_rf_9 = run_experiment(data_perm_9, default_rfc)
    #results_rf_9 = run_experiment(preprocessor.get_matrix_split(['sex', 'age', 'survived']), default_rfc)
    #
    # print("\nrf perm 10")
    # results_rf_10 = run_experiment(data_perm_10, default_rfc)
    #
    # print("\nperm 1 warm")
    # results_rf_1_warm = run_experiment(data_perm_1, warmstart_rfc)
    #
    # print("\nperm 1 gradient boost forest")
    # gradient_boost_rf = run_experiment(data_perm_1, gradient_boost)
    #
    # print("\nperm 1 random")
    # results_rf_1_random = run_experiment(data_perm_1, random_rfc)
    #
    # print("\nperm 1 est")
    # results_rf_1_perm = run_experiment(data_perm_1, est_rfc)
    #
    # print("\nperm 1 depth")
    # results_rf_1_depth = run_experiment(data_perm_1, depth_rfc)
    #
    # print("\nperm 1 combo 1")
    # results_rf_1_combo_1 = run_experiment(data_perm_1, combo_1_rfc)
    # print("\ncombo exp perm 1")
    #
    # print("\nperm 5 combo 1")
    # results_rf_5_combo_1 = run_experiment(data_perm_5, combo_1_rfc)
    #
    # print("\nperm 9 combo 1")
    # results_rf_9_combo_1 = run_experiment(data_perm_9, combo_1_rfc)
    #
    # print("\nperm 10 combo 1")
    # results_rf_10_combo_1 = run_experiment(data_perm_10, combo_1_rfc)
    #
    # #TODO: use perm 4 to try to optimize for best results
    # print("\nperm 4 combo 1")
    # results_rf_4_combo_1 = run_experiment(data_perm_4, combo_1_rfc)
    #
    # print("\n extreme random")
    # results_rf_1_ext = run_experiment(data_perm_1, extreme_rand_rfc)
    #
    # print("\n ")
    # pprint.pprint(experiment_result_dicts)


#TODO: modify to write results to file
def run_experiment(data_perm, rfc):
    """

    :param data_perm:
    :param rfc:
    :return: dict containing experiment results
    """
    dtr = EnsembleRunner(data_perm, rfc)
    dtr.run()
    dtr.print_feature_importance()
    dtr.print_accuracy()
    return dtr.prediction_results, dtr.accuracy
# compared failed predictions, are there any outliers who are consistent across all models?

if __name__ == '__main__':
    main()
