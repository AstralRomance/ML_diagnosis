import pandas as pd

'''
class take excel table and make csv dataset with formalized data 

__init__() method
parsed_file - path to excel file for receive data
age_group_matrix - matrix for using as patients age groups; -1 -1 -1 for empty age cells
sex_matrix - matrix for using as sex; -1 -1 for empty age cells 
'''


class DataPreparer:
    def __init__(self, parsed_file):
        self.age_group_matrix = ['<0', '0-17', '17-21', '21-55', '55-75', '75-90', '>90']
        self.to_parse = parsed_file
        self.patient_id = None
        self.dataset_unmodified = None
        self.dataset_no_useless = None

#   reading file with parsed_file path and converts to csv
#   dataset with only useful parameters with the next data formalizing
#   return formed dataset
    def parse(self):
        data_xls = pd.read_excel(self.to_parse, 'part1', index_col=0)
        data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')

        output_dataset = pd.read_csv('res_dataset.csv', sep=';')
        output_dataset = output_dataset.fillna(-1)
        self.dataset_unmodified = output_dataset
        self.dataset_no_useless = self.dataset_unmodified

    def remove_useless(self, useless_fields=None):
        if useless_fields:
            for i in useless_fields:
                self.dataset_no_useless = self.dataset_no_useless.drop(i, 1)
            temp = 0
            for columns in self.dataset_no_useless:
                for val in self.dataset_no_useless[columns]:
                    if val == -1:
                        temp += 1
                if temp/len(self.dataset_no_useless[columns]) >= 0.5:
                    self.dataset_no_useless = self.dataset_no_useless.drop(columns, 1)
                temp = 0
            for i in self.dataset_no_useless:
                for val in self.dataset_no_useless[i]:
                    if (type(val) == str) and (('>' in val) or ('<' in val)):
                        self.dataset_no_useless.loc[self.dataset_no_useless[i] == val, i] = val[1::]
        else:
            print('Nothing to delete')
            return 0

    def dataset_to_numeric(self):
        for col in self.dataset_no_useless.columns:
            self.dataset_no_useless[col] = pd.to_numeric(self.dataset_no_useless[col], errors='ignore', downcast='float')

    def gender_changes(self, gender_column):
        if gender_column:
            temp = []
            self.dataset_no_useless[gender_column] = self.dataset_no_useless[gender_column].map(
                {'Мужской': 0, 'Женский': 1}
            )
        else:
            print('Nothing to delete')
            return 0

    def replace_to_BMI(self, w_h_columns=None):
        if w_h_columns:
            temp = []
            for i, vals in enumerate(self.dataset_no_useless[w_h_columns[0]].values):
                if float(self.dataset_no_useless[w_h_columns[1]][i]) == 0.0 or float(self.dataset_no_useless[w_h_columns[1]][i]) == -1.0:
                    temp.append(0)
                else:
                    temp.append(float(self.dataset_no_useless[w_h_columns[0]][i]) /
                                (float(self.dataset_no_useless[w_h_columns[1]][i]) / 100) ** 2)
            for cols in w_h_columns:
                self.dataset_no_useless = self.dataset_no_useless.drop(cols, 1)
            self.dataset_no_useless = self.dataset_no_useless.assign(BMI=temp)
        else:
            print('Nothing to delete')
            return 0

    def ages_change(self, ages_column=None):
        if ages_column:
            self.dataset_no_useless['<0'] = [1 if i in list(range(-1, 0)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless['0-17'] = [1 if i in list(range(0, 17)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless['17-21'] = [1 if i in list(range(17, 21)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless['21-55'] = [1 if i in list(range(21, 55)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless['55-75'] = [1 if i in list(range(55, 75)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless['75-90'] = [1 if i in list(range(75, 90)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless['>90'] = [1 if i in list(range(90, 1000)) else 0 for i in self.dataset_no_useless[ages_column]]
            self.dataset_no_useless = self.dataset_no_useless.drop(ages_column, 1)
        else:
            print('Nothing to delete')
            return 0

#   change source file
    def change_parsed_file(self, file_path):
        self.to_parse = file_path

    @property
    def get_dataset_no_useless(self):
        return self.dataset_no_useless

    @property
    def get_dataset_unmodified(self):
        return self.dataset_unmodified
