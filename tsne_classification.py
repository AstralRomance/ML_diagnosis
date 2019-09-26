from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import xls_to_csv_parser


class TsneTest():
    def __init__(self):
        prs = xls_to_csv_parser.Parser()
        self.data = prs.parser()

    def start_classification(self):
        for i in range(50, 1000, 50):
            for j in range(10, 100, 10):
                classifier = TSNE(learning_rate=i, perplexity=j)
                classification = classifier.fit_transform(self.data)
                self.__make_tsne_plot(classification, i, j)

    def __make_tsne_plot(self, classification, lr, prp):
        fig = plt.figure()
        plt.scatter(classification[:, 0], classification[:, 1], c=self.data['Пол'].map({0: 'b', 1: 'r', 2: 'g'}))
        print(str(lr) + ' rate done')
        fig.savefig('graphs/tsne/' + str(lr) + str(prp) + 'lernR.png')
        plt.close(fig=fig)
