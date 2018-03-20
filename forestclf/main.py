from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from forestclf.ensemblerunner import EnsembleRunner
import pprint
import itertools as iter
import sys


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
    
    classifier_dict = {
        "default_rfc": default_rfc,
        # "warmstart_rfc": warmstart_rfc,
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
    high_performers = dict()
    high_perf_count = 0
    experiment_failed_results = []
    
    text = open("../forestclf/failedresults.txt", 'w')
    
    experiments_tried_map = dict
    
    # TODO: to make more efficient -- scale features just once instead of for each feature set-- need to rework preprocessor
    ## ^ will be done over spring break ##
    
    if sys.argv.__contains__("-brute"):
        for k in classifier_dict:
            for i in range(len(feature_list)):
                if i < 1:
                    continue
                for j in iter.combinations(feature_list, i):
                    # create hash key to avoid running same experiments with diff orders of same feature

                    j = list(j)
                    j.append(survived)
                    data = preprocessor.get_matrix_split(list(j))
                    pred_results, acc, failed_results = run_experiment(data, classifier_dict[k])
                    if acc > max_dict["value"]:
                        max_dict["value"] = acc
                        max_dict["feature_list"] = j
                        max_dict["classifier"] = k
                    
                    if acc > 0.77:
                        high_performers[high_perf_count] = {
                            "value": acc,
                            "feature_list": j,
                            "classifier": k
                        }
                        experiment_failed_results[high_performers[high_perf_count]] = failed_results
                        high_perf_count += 1
    
    print("MAX:")
    print(max_dict)
    print(" # of High performers: ", high_perf_count)
    # pprint.pprint(high_performers)
    
    #
    # #non biased feature list -- 77.21% accuracy  -- classifier default rfc-
    nb_feature_list = preprocessor.get_matrix_split(['sex', 'age', 'pclass', 'parch', 'survived'])
    pred_results, acc, failed_results = run_experiment(nb_feature_list, default_rfc)
    experiment_failed_results.append(failed_results)

    # # 80.15% accuracy -- boost
    opt_feature_list = preprocessor.get_matrix_split(['name', 'embarked', 'survived'])
    pred_results, acc, failed_results = run_experiment(opt_feature_list, default_rfc)
    experiment_failed_results.append(failed_results)

    feature_list = preprocessor.get_matrix_split(['sex', 'age', 'ticket', 'fare', 'pclass', 'name', 'sibsp', 'parch',
                                                  'embarked', 'survived'])
    pred_results, acc, failed_results = run_experiment(feature_list,
                                                       default_rfc,
                                                       create_graph=False,
                                                       print_confusion_matrix=True)
    experiment_failed_results.append(failed_results)

    # # pprint.pprint(experiment_failed_results) #TODO: keep or remove


def run_experiment(data_perm, rfc, create_graph = False, print_confusion_matrix = False):
    """

    :param data_perm:
    :param rfc:
    :return: dict containing experiment results
    """
    dtr = EnsembleRunner(data_perm, rfc)
    print("running experiment for ", rfc)
    dtr.run()
    dtr.print_feature_importance()
    dtr.print_accuracy()
    if create_graph is True:
        dtr.graph_results()

    if print_confusion_matrix is True:
        print(dtr.confusion_matrix)
    print("\n")
    return dtr.prediction_results, dtr.accuracy, dtr.failed_predictions


# compared failed predictions, are there any outliers who are consistent across all models?

def get_hash(feature_list, k):
    """
    
    :param feature_list:
    :param k: name of classifer
    :return:
    """
    hash_key = 0;
    for feature in range(len(feature_list)):
        hash_key += hash(feature_list[feature])
    hash_key = hash_key + hash(k)
    print("\nhash key for classifier ", k, " and list ", feature_list, " : ", hash_key) #TODO: remove
    return hash_key

if __name__ == '__main__':
    main()
