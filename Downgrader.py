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

    @property
    def get_unmodified_data(self):
        return self.data
