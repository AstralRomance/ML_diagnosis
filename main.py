import seaborn as sns
from matplotlib import numpy as np

from Parser import DataPreparer
from CustomConsoleInterface import CustomConsoleInterface
from Downgrader import Downgrader
from Classifier import KMeansClassifier
from Visualizer import Visualizer

interface = CustomConsoleInterface()
vis = Visualizer()
dp = DataPreparer('dataset.xlsx')

dp.parse()
dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose useless', 'useless_columns').values())
dp.gender_changes(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                           'choose gender column', 'gender_column').values())
dp.ages_change(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                        'choose age column', 'age_column').values())
dp.replace_to_BMI(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose weight and height columns (following is important)',
                                               'BMI_replaceing').values())
dp.dataset_to_numeric()
print(dp.get_dataset_no_useless)
vis.make_heatmap(dp.get_dataset_no_useless, dp.get_ages)

if 'clustering' in interface.make_list([{'name': 'clustering'}, {'name': 'classification'}], 'Choose analysis mode',
                                       'analysis_mode').values():
    kmeans = KMeansClassifier(dp.get_dataset_no_useless, 500)
    kmeans.train(max_iter=550)
    kmeans.predict()
    print('pairplot building')
    print(kmeans.get_clustered[kmeans.get_clustered['clusters'] == 2])
    #kmeans_correct = kmeans.get_clustered[kmeans.get_clustered['clusters'] == 0]
    vis.make_pairplot(kmeans.get_clustered, dp.get_ages, f'{3}_cluster')


else:
    val_col = interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                      'choose valid columns', 'valid_columns').values()
    for train_length in range(100, 700, 50):
        for n_clusters in range(3, 6):
            for max_iter in range(500, 4000, 200):
                kmeans = KMeansClassifier(dp.get_dataset_no_useless, train_length)
                kmeans.choose_clustering_columns(*val_col)
                kmeans.train(n_clusters, max_iter)
                vis.make_pairplot(kmeans.get_test, *dp.get_ages, kmeans.predict(), (train_length, n_clusters ))
