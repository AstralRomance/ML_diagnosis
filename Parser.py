import pandas as pd

'''
class take excel table and make csv dataset with formalized data 

__init__() method
parsed_file - path to excel file for receive data
age_group_matrix - matrix for using as patients age groups; -1 -1 -1 for empty age cells
sex_matrix - matrix for using as sex; -1 -1 for empty age cells 
'''


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
        self.patient_id = None
        self.dataset_unmodified = None

#   reading file with parsed_file path and converts to csv
#   dataset with only useful parameters with the next data formalizing
#   return formed dataset
    def parse(self):
        data_xls = pd.read_excel(self.to_parse, 'part1', index_col=0)
        data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')

        output_dataset = pd.read_csv('res_dataset.csv', sep=';')
        output_dataset = output_dataset.fillna(-1)
        self.dataset_unmodified = output_dataset

    def remove_useless(self, dataset, useless_fields):
        pass

#   change source file
    def change_parsed_file(self, file_path):
        self.to_parse = file_path


#   replace some parameters for matrix rows or calculated values
    def __simple_formalizing(self, dataset):
        imt = []
        for counter, val in enumerate(dataset['VES'].values):
            if val == 0 or dataset['ROST'].values[counter] == 0:
                imt.append(-1)
            else:
                imt.append(dataset['VES'].values[counter]/(dataset['ROST'].values[counter]/100)**2)

        for counter, sex in enumerate(dataset['Пол'].values):
            if sex == 'Мужской':
                dataset['Пол'].values[counter] = 1
            if sex == 'Женский':
                dataset['Пол'].values[counter] = 2

        dataset = dataset.assign(IMT=imt)
        dataset = dataset.drop('ROST', 1)
        dataset = dataset.drop('VES', 1)
        dataset = dataset.drop('IND', 1)

        dataset = dataset.fillna(-1)

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

        #   debug info, check for correct changes in dataset
        with open('temp2.txt', 'w') as tm:
            for i in dataset.values:
                for j in i:
                    tm.write(str(j) + ' ')
                tm.write('\n')

    @property
    def get_dataset_unmodified(self):
        return self.dataset_unmodified
