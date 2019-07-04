from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import xls_to_csv_parser


classifier = TSNE(learning_rate=1000, random_state=17)

data = xls_to_csv_parser.parser()


classification = classifier.fit_transform(data)
#print(classification.labels_)
plt.scatter(classification[:, 0], classification[:, 1], c=data['Пол'].map({0: 'b', 1:'r', 2:'g'}))

plt.show()