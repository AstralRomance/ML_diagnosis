from tsne_classification import *
from kmeans_classification import *

tsn = TsneTest()
classifier = KMeans_make(tsn.start_modeling())
classifier.classification_start()
