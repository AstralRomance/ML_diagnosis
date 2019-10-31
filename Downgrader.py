from sklearn.manifold import TSNE

'''
Class downgrades space to Euclidean. Dataset as input info.
'''


class Downgrader():
    def downgrade(self, dataset, learning_rate, perplexity):
        classifier = TSNE(learning_rate=learning_rate, perplexity=perplexity)
        classification = classifier.fit_transform(dataset)
        return classification

