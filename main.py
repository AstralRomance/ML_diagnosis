from matplotlib import pyplot as plt
import pandas as pd

from Parser import DataPreparer
from CustomConsoleInterface import CustomConsoleInterface
from Analyzer import Analyzer
from Classifier import KMeansClassifier, BayesClassifier, Forest
from Visualizer import Visualizer

interface = CustomConsoleInterface()
vis = Visualizer()
dp = DataPreparer('sources/dataset.xlsx')
analyzer = Analyzer()
dp.parse()
print(dp.get_dataset_no_useless)

additional_info = False

dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose useless', 'useless_columns').values())
if additional_info:
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
'''
Temporary numeric mode. Make menu for choice later. Coerce for another strange dataset
Use coerce for ONLY numeric or already encoded data
'''
dp.dataset_to_numeric('coerce')
#vis.make_heatmap(dp.get_dataset_no_useless, dp.get_ages)
pairplot_flag = True

if 'clustering' in interface.make_list([{'name': 'clustering'}, {'name': 'classification'}], 'Choose analysis mode',
                                       'analysis_mode').values():
    print(dp.get_dataset_no_useless)
    metric_collection = []
    for train_l in range(3000, 10000, 250):
        kmeans = KMeansClassifier(dp.get_dataset_no_useless, train_l)
        for n_clusters in range(3, 30):
            for m_iter in range(500, 1500, 200):
                kmeans.train(clusters=n_clusters, max_iter=m_iter)
                kmeans.predict()
                metric_collection.append(kmeans.metrics)
                if pairplot_flag:
                    try:
                        vis.make_pairplot(kmeans.get_clustered, dp.get_ages, f'{train_l}_trainL_{n_clusters}_clusters_{m_iter}_learning_rate')
                    except Exception as e:
                        print(f'{e} has been dropped')
    analyzer.metric_collection('KMeans', metric_collection)
    analyzer.best_clustering_find()
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

