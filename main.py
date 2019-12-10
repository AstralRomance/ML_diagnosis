from matplotlib import pyplot as plt
import pandas as pd

from Parser import DataPreparer
from CustomConsoleInterface import CustomConsoleInterface
from Downgrader import Downgrader
from Analyzer import Analyzer
from Classifier import KMeansClassifier, BayesClassifier, Forest
from Visualizer import Visualizer

interface = CustomConsoleInterface()
vis = Visualizer()
dp = DataPreparer('sources/dataset.xlsx')
analyzer = Analyzer()
dp.parse()
print(dp.get_dataset_no_useless)
dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose useless', 'useless_columns').values())
dp.gender_changes(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                           'choose gender column', 'gender_column').values())
dp.ages_change(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                        'choose age column', 'age_column').values())
dp.replace_to_BMI(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose weight and height columns (following is important)',
                                               'BMI_replaceing').values())
dp.invalid_check(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'Choose places to invalid checking',
                                               'Invalid check').values())

dp.dataset_to_numeric()
print(dp.get_dataset_no_useless.columns)
#vis.make_heatmap(dp.get_dataset_no_useless, dp.get_ages)

if 'clustering' in interface.make_list([{'name': 'clustering'}, {'name': 'classification'}], 'Choose analysis mode',
                                       'analysis_mode').values():
    kmeans = KMeansClassifier(dp.get_dataset_no_useless, 500)
    for i in range(500, 5000, 150):
        kmeans.train(max_iter=1500)
        kmeans.predict()
        try:
            vis.make_pairplot(kmeans.get_clustered, dp.get_ages, f'{i}_cluster')
        except Exception as e:
            print(f'{e} has been dropped')
        for j in set(kmeans.get_clustered['clusters']):
            analyzer.probability_per_cluster(len(kmeans.get_clustered[kmeans.get_clustered['clusters'] == j]), len(kmeans.get_test), j)
else:
    train_score = []
    test_score = []
    class_label = dp.make_class_labels(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                                 'choose classes labels', 'class_labels').values())
    if 'Random Forest' in interface.make_list([{'name': 'Bayes'}, {'name': 'Random Forest'}], 'Choose classifier',
                                              'classifier').values():
        t = []
        for train_len in range(100, 1400, 50):
            forest = Forest(dp.get_dataset_no_useless, train_len, class_label)
            t.append(forest.train())
        res_df = pd.DataFrame(t, columns=[i for i in dp.dataset_no_useless.drop(class_label, 1).keys()])
        print(res_df)
    else:
        for train_len in range(100, 1401, 50):
            bayes = BayesClassifier(dp.get_dataset_no_useless, train_len, class_label)
            train_score.append(bayes.train())
            test_score.append(bayes.predict())
            print('confusion matrix')
            vis.make_confusion_matrix(bayes.get_test_labels, bayes.get_prediction, train_len)
        print(train_score)
        print(test_score)
        vis.make_overlearn_check_plot(train_score, test_score, range(100, 1401, 50), 'bayes')

