# Импортируем библиотеки
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import xls_to_csv_parser

dbscan = DBSCAN(eps=100)
data = xls_to_csv_parser.parser()
# Обучаем
dbscan.fit(data)

# Уменьшаем размерность при помощи метода главных компонент
pca = PCA(n_components=2).fit(data)
pca_2d = pca.transform(data)
for i in range(len(dbscan.labels_)):
    if dbscan.labels_[i] != -1:
        print(dbscan.labels_[i])

plot_lst1 = []
plot_lst2 = []
plot_lst3 = []
plot_lst4 = []

# Строим в соответствии с тремя классами
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        #plot_lst1.append(pca_2d[i, 0], pca_2d[i, 1])
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        #plot_lst2.append(pca_2d[i, 0]), pca_2d[i, 1])
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='x')
    else:
        c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c ='m', marker='*')




plt.show()