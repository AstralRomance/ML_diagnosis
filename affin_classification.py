from sklearn.cluster import AffinityPropagation
import numpy as np
import xls_to_csv_parser
import matplotlib.pyplot as plt
from itertools import cycle

data = xls_to_csv_parser.parser()

classifier = AffinityPropagation().fit(data)
