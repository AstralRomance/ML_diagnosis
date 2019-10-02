from sklearn.manifold import TSNE

'''
Class downgrades space to Euclidean. Dataset as input info.
'''


class Downgrader():
    def __init__(self, dataset):
        self.data = dataset

    def downgrade(self, learning_rate, perplexity):
        classifier = TSNE(learning_rate=learning_rate, perplexity=perplexity)
        classification = classifier.fit_transform(self.data)
        return classification

'''
class TsneTest():
    def __init__(self):
        prs = xls_to_csv_parser.Parser('datasetxls.xlsx')
        self.data = prs.parser()

    def test_modeling(self):
        for i in range(50, 1000, 50):
            for j in range(10, 100, 10):
                classifier = TSNE(learning_rate=i, perplexity=j)
                classification = classifier.fit_transform(self.data)

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

'''