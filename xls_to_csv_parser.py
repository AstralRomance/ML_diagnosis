import pandas as pd
from sklearn.preprocessing import Normalizer
from matplotlib import numpy as np


class Parser:
    def __init__(self, parsed_file):
        self.age_group_matrix = [
            [-1, -1, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1]
        ]
        self.sex_matrix = [
            [-1, -1],
            [0, 0],
            [0, 1]
        ]
        self.to_parse = parsed_file

    def parser(self):
        data_xls = pd.read_excel(self.to_parse, 'part1', index_col=0)
        data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')

        output_dataset = pd.read_csv('res_dataset.csv', sep=';')
        output_dataset = output_dataset.drop('№', 1)
        output_dataset = output_dataset.drop('%Код экземпляра', 1)

        output_dataset = output_dataset.fillna(-1)
        self.simple_normalizing(output_dataset)
        return output_dataset

    def simple_normalizing(self, dataset):
        with open('temp1.txt', 'w') as temp:
            for i in dataset['Пол'].values:
                temp.write(str(i))
        for i in range(len(dataset['Пол'].values)):
            if dataset['Пол'][i] == 'Мужской':
                dataset['Пол'].values[i] = 1
            if dataset['Пол'][i] == 'Женский':
                dataset['Пол'].values[i] = 2


        temp = []
        for j in dataset['Возраст'].values:
            if j < 0.0:
                temp.append(self.age_group_matrix[0])

            if j < 17.0 and j > 0.0:
                temp.append(self.age_group_matrix[1])

            if j >= 17.0 and j < 21.0:
                temp.append(self.age_group_matrix[2])

            if j >= 21.0 and j < 55.0:
                temp.append(self.age_group_matrix[3])

            if j >= 55.0 and j < 75.0:
                temp.append(self.age_group_matrix[4])

            if j >= 75.0 and j < 90.0:
                temp.append(self.age_group_matrix[5])

            if j >= 90.0:
                temp.append(self.age_group_matrix[6])
        dataset = dataset.assign(Возрастная_группа=temp)
        dataset = dataset.drop('Возраст', 1)
        with open('temp2.txt', 'w') as tm:
            for i in dataset.values:
                for j in i:
                    tm.write(str(j) + ' ')
                tm.write('\n')

    def change_parsed_file(self, file_path):
        self.to_parse = file_path