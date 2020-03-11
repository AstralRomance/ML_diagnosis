from Parser import DataPreparer
from CustomConsoleInterface import CustomConsoleInterface
from Analyzer import Analyzer
from Classifier import KMeansClassifier, BayesClassifier, Forest
from Visualizer import Visualizer
from setup import Setup
import logging
import warnings

logging.basicConfig(filename='debug/debug.log', level=logging.DEBUG)
warnings.filterwarnings('error')
setup = Setup()

interface = CustomConsoleInterface()
vis = Visualizer()
dp = DataPreparer('sources/dataset.xlsx')
analyzer = Analyzer()

dp.parse()
additional_info = True
pairplot_flag = True
test_flag = False
separate_flag = True

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


analysis_mode = interface.make_list([{'name': 'clustering'}, {'name': 'classification'}, {'name': 'custom_classification'}],
                                    'Choose analysis mode',
                                    'analysis_mode').values()
if 'clustering' in analysis_mode:
    '''
    Temporary numeric mode. Make menu for choice later. Coerce for another strange dataset
    Use coerce for ONLY numeric or already encoded data
    '''
    #Flag is True if analysis need diagnosis separating

    if separate_flag:
        dp.separate_class_labels(*interface.make_checkbox(
            [{'name': i} for i in dp.get_dataset_no_useless.keys()],
            'Choice class labels',
            'classes:').values())
        dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                                   'choose useless', 'useless_columns').values())
    dp.dataset_to_numeric('coerce')
    vis.make_heatmap(dp.get_dataset_no_useless, dp.get_ages, 'all_features', 'Not clustered')
    if test_flag:
        print(dp.get_dataset_no_useless)
        metric_collection = []
        for train_l in range(int(len(dp.get_dataset_no_useless)*0.3),
                             int(len(dp.get_dataset_no_useless)*0.8),
                             int(len(dp.get_dataset_no_useless)*0.05)):
            kmeans = KMeansClassifier(dp.get_dataset_no_useless, train_l)
            for n_clusters in range(3, 30):
                for m_iter in range(500, 1500, 200):
                    kmeans.train(clusters=n_clusters, max_iter=m_iter)
                    kmeans.predict()
                    metric_collection.append(kmeans.metrics)
                    test = kmeans.get_clustered
                if pairplot_flag:
                    try:
                        vis.make_pairplot(kmeans.get_clustered, dp.get_ages, f'{train_l}_trainL_{n_clusters}_clusters')
                        print(f'pairplot for train distribution {train_l} clusters {n_clusters} built successfully')
                    except Exception as e:
                        print(f'{e} has been dropped')
        analyzer.metric_collection('KMeans', metric_collection)
        analyzer.best_clustering_find()
    else:
        kmeans_best = KMeansClassifier(dp.dataset_no_useless, 15000)
        kmeans_best.train(3, 900)
        kmeans_best.predict()
        kmeans_best.clusters_to_excel()
        analyzer.separate_clusters(kmeans_best.get_clustered)
        forest_test_scores = []
        forest_train_scores = []
        train_l_list = [i for i in range(int(len(kmeans_best.get_clustered) * 0.2),
                                             int(len(kmeans_best.get_clustered) * 0.8), 100)]
        for train_l in train_l_list:
            forest = Forest(kmeans_best.get_clustered, 'clusters', train_l)
            forest.train()
            forest.predict()
            analyzer.make_features_rate(forest.get_feature_importances, kmeans_best.get_data.columns, train_l)
            forest_test_scores.append(forest.collect_test_score('cross_validate'))
            forest_train_scores.append(forest.collect_train_score('cross_validate'))
        analyzer.probability_per_cluster(kmeans_best.get_test)
        analyzer.normal_check()
        analyzer.calc_predictors_interval()
        test_len_list = [len(dp.get_dataset_no_useless) - i for i in train_l_list]
        #for i in range(len(forest_test_scores[0])):
        #    vis.make_overlearn_check_plot([k for k in forest_train_scores[:, i]], [k for k in forest_test_scores[:, i]], train_l_list,
        #                                  f'forest_test/random_forest_for_best_clustering{i}')
        #vis.make_compare_plot(test_len_list, forest_test_scores, 'forest_test/random_forest_for_best_clustering_test')
        #vis.make_compare_plot(train_l_list, forest_train_scores, 'forest_test/random_forest_for_best_clustering_train')

        for counter, cluster in enumerate(analyzer.separated_clusters):
            try:
                for predictor in cluster:
                    vis.distribution_hist(cluster[predictor], counter, predictor)
            except Warning as e:
                print(f'Found {e} Warning check this')
                continue
            vis.make_heatmap(cluster, dp.get_ages, 'cluster_number', counter)

elif 'classification' in analysis_mode:
    dp.separate_class_labels(*interface.make_checkbox(
                                            [{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                            'Choice class labels',
                                            'classes:').values())
    for train_l in range(int(len(dp.get_dataset_no_useless) * 0.4), int(len(dp.get_dataset_no_useless)*0.8),
                         int(len(dp.get_dataset_no_useless)*0.05)):
        forest = Forest(dp.get_dataset_no_useless, 'clusters', train_l)
        forest.train()
        print(forest.predict())
    dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose useless', 'useless_columns').values())
    print(dp.get_dataset_no_useless)


