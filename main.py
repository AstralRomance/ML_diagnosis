from sklearn.cluster import KMeans

from Parser import DataPreparer
from CustomConsoleInterface import CustomConsoleInterface
from Downgrader import Downgrader
from Classifier import KMeansClassifier

interface = CustomConsoleInterface()

dp = DataPreparer('dataset.xlsx')
dp.parse()

dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose useless', 'useless_columns').values())
dp.gender_changes(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                           'choose gender column', 'gender_column').values())
dp.ages_change(*interface.make_list([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                        'choose age column', 'age_column').values())
dp.remove_useless(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose useless', 'useless_columns').values())
dp.replace_to_BMI(*interface.make_checkbox([{'name': i} for i in dp.get_dataset_no_useless.keys()],
                                               'choose weight and height columns (following is important)',
                                               'BMI_replaceing').values())
dp.dataset_to_numeric()


print(dp.get_dataset_no_useless)
kmeans = KMeansClassifier(dp.get_dataset_no_useless, 500)
kmeans.train()
kmeans.predict()
