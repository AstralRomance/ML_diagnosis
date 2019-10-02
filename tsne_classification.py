from sklearn.manifold import TSNE
from sklearn.manifold import locally_linear_embedding
import matplotlib.pyplot as plt

import xls_to_csv_parser

class TsneTest():
    def __init__(self):
        prs = xls_to_csv_parser.Parser('datasetxls.xlsx')
        self.data = prs.parser()
        self.manifold_data = []
        #prs.change_parsed_file('train.xlsx')
        #self.training = prs.parser()

    def test_modeling(self):
        for i in range(50, 1000, 50):
            for j in range(10, 100, 10):
                classifier = TSNE(learning_rate=i, perplexity=j)
                classification = classifier.fit_transform(self.data)

                #print(locally_linear_embedding(self.data, n_neighbors=j, n_components=36))

                self.__make_tsne_plot(classification, i, j)

    def start_modeling(self):
        classifier = TSNE(learning_rate=500, perplexity=50)
        classification = classifier.fit_transform(self.data)
        return classification

    def __make_tsne_plot(self, classification, lr, prp):
        fig = plt.figure()
        plt.scatter(classification[:, 0], classification[:, 1], c=self.data['Пол'].map({-1: 'yellow', 1: 'red', 2: 'green'}))
        print(str(lr) + ' rate done')
        #plt.show()
        fig.savefig('graphs/tsne/' + str(lr) + str(prp) + 'lernR.png')
        plt.close(fig=fig)

