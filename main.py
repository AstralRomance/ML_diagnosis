from Parser import DataPreparer
from CustomConsoleInterface import CustomConsoleInterface
from Downgrader import Downgrader
from Classifier import KMeansClassifier, BayesClassifier
from Visualizer import Visualizer

interface = CustomConsoleInterface()
vis = Visualizer()
dp = DataPreparer('sources/dataset.xlsx')

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
print(dp.get_dataset_no_useless)
#vis.make_heatmap(dp.get_dataset_no_useless, dp.get_ages)

if 'clustering' in interface.make_list([{'name': 'clustering'}, {'name': 'classification'}], 'Choose analysis mode',
                                       'analysis_mode').values():
    kmeans = KMeansClassifier(dp.get_dataset_no_useless, 500)
    kmeans.train(max_iter=550)
    kmeans.predict()
    print('pairplot building')
    for i in set(kmeans.get_clustered['clusters']):
        try:
            vis.make_pairplot(kmeans.get_clustered[kmeans.get_clustered['clusters'] == i], dp.get_ages, f'{3}_cluster')
        except Exception as e:
            print(f'{e} has been dropped')
            continue
else:
    train_score = []
    test_score = []
    class_label = dp.make_class_labels(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                                 'choose classes labels', 'class_labels').values())
    for train_len in range(100, 1400, 50):
        bayes = BayesClassifier(dp.get_dataset_no_useless, train_len, class_label)
        train_score.append(bayes.train())
        test_score.append(bayes.predict())
    print(train_score)
    print(test_score)
    vis.make_overlearn_check_plot(train_score, test_score, range(100, 1400, 50), 'bayes')
