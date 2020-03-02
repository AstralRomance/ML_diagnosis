import os

import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle

'''
class take excel table and make csv dataset with formalized data 

__init__() method
parsed_file - path to excel file for receive data
age_group_matrix - matrix for using as patients age groups; -1 -1 -1 for empty age cells
sex_matrix - matrix for using as sex; -1 -1 for empty age cells 
'''

class DataPreparer:
    def __init__(self, parsed_file):
        '''
        :param parsed_file: excel table with data
        '''
        self.age_intervals = [(-1, 0), (0, 17), (17, 21), (21, 55), (55, 75), (75, 90), (90, 1000)]
        self.to_parse = parsed_file
        self.encoder = preprocessing.LabelEncoder()
        self.patient_id = None
        self.dataset_unmodified = None
        self.dataset_no_useless = None

    def parse(self):
        '''
            Form dataframe from excel table. Make csv for safety
        :return: None
        '''
        if 'res_dataset.csv' in os.listdir(os.getcwd()):
            output_dataset = pd.read_csv('res_dataset.csv', sep=';')
            print('Use csv as a source')
        else:
            print('Start excel reading')
            data_xls = pd.read_excel(self.to_parse, index_col=0)
            data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')
            output_dataset = pd.read_csv('res_dataset.csv', sep=';')
            output_dataset = shuffle(output_dataset)
            print('Formed csv source')
        output_dataset = output_dataset.fillna(-1)
        self.dataset_unmodified = output_dataset
        self.dataset_no_useless = self.dataset_unmodified
        print(self.dataset_no_useless.replace({'N':None}))

    def remove_useless(self, useless_fields=None):
        '''
            Remove useless columns from dataframe
        :param useless_fields: list of dataframe useless columns
        :return: 0 if useless columns list is empty
        '''
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
        else:
            print('Nothing to delete')
            return 0

    def replace_undef_symbols(self, columns):
        '''
        :param columns: list of columns for replace symbols like < > in string type predictors
        :return:
        '''
        for i in columns:
            for val in self.dataset_no_useless[i]:
                if (type(val) == str) and (('>' in val) or ('<' in val)):
                    self.dataset_no_useless.loc[self.dataset_no_useless[i] == val, i] = val[1::]

    def dataset_to_numeric(self, numeric_mode=None):
        '''
            Converting data to numeric format
        :param numeric_mode: errors parameter for pandas to_numeric method
        :return: None
        '''
        for col in self.dataset_no_useless.columns:
            self.dataset_no_useless[col] = pd.to_numeric(self.dataset_no_useless[col], errors=numeric_mode, downcast='float')
        self.dataset_no_useless.fillna(-1)

    def gender_changes(self, gender_column=None):
        '''
            Encode column, chosen as "gender"
        :param gender_column: label of gender column
        :return: 0 if gender_column is None
        '''
        if gender_column:
            self.dataset_no_useless[gender_column] = self.dataset_no_useless[gender_column].map(
                {'Мужской': 0, 'Женский': 1}
            )
        else:
            print('Nothing to replace')
            return 0

    def replace_to_BMI(self, w_h_columns=None):
        '''
            Replace columns, chosen as "weight" and "height" to BMI
        :param w_h_columns: weight and height columns labels
        :return: 0 if columns are empty
        '''
        if w_h_columns:
            weight_bmi = self.dataset_no_useless[w_h_columns[0]].copy()
            height_bmi = self.dataset_no_useless[w_h_columns[1]].copy()
            for cols in w_h_columns:
                self.dataset_no_useless = self.dataset_no_useless.drop(cols, 1)
            bmi = []
            for count, val in enumerate(height_bmi):
                if (height_bmi[count] == 0) or (weight_bmi[count] == 0):
                    bmi.append(-1)
                else:
                    bmi.append(weight_bmi[count]/((height_bmi[count]/100)**2))
            self.dataset_no_useless['BMI'] = bmi
            self.dataset_no_useless = self.dataset_no_useless.drop(self.dataset_no_useless[self.dataset_no_useless.BMI > 300].index)
        else:
            print('Nothing to replace')
            return 0

    def ages_change(self, ages_column=None):
        '''
            Replace column, chosen as "age" to age group matrix
        :param ages_column: age column label.
        :return: 0 if column is empty
        '''
        if ages_column:
            for i in self.age_intervals:
                for j in self.dataset_no_useless[ages_column]:
                    if j in range(*i):
                        self.dataset_no_useless[str(i)] = 1
                    else:
                        self.dataset_no_useless[str(i)] = 0
            self.dataset_no_useless = self.dataset_no_useless.drop(ages_column, 1)
        else:
            print('Nothing to replace')
            return 0

    def invalid_check(self, invalid_columns=None):
        '''
            Check for correct values in chosen columns
        :param invalid_columns: list of labels of columns to check
        :return: 0 if list if empty
        '''
        for col in invalid_columns:
            self.dataset_no_useless[col].replace(0.0, -1, inplace=True)
        for col in invalid_columns:
            print(self.dataset_no_useless[col])
        else:
            print('Nothing to replace')
            return 0

    def separate_class_labels(self, labels):
        max_len_main_diag = 0
        max_len_additional_diag = 0
        main_diag = True
        main_diag_list = []
        additional_diag_list = []
        for label in labels:
            for val in self.dataset_no_useless[label]:
                if main_diag:
                    try:
                        if len(val.split('>>')) > max_len_main_diag:
                            max_len_main_diag = len(val.split('>>'))
                        main_diag_list.append(val.split('>>'))
                    except Exception as e:
                        main_diag_list.append([-1, ])
                else:
                    try:
                        if len(val.split('>>')) > max_len_additional_diag:
                            max_len_additional_diag = len(val.split('>>'))
                        additional_diag_list.append(val.split('>>'))
                    except Exception as e:
                        additional_diag_list.append([-1, ])
            main_diag = not main_diag
        main_diag_df = pd.DataFrame(main_diag_list, columns=[f'Main_diag_{i}'
                                                             for i in range(max_len_main_diag)]).fillna(-1)
        add_diag_df = pd.DataFrame(additional_diag_list, columns=[f'Add_diag_{i}'
                                                                  for i in range(max_len_additional_diag)]).fillna(-1)

        for frame in (main_diag_df, add_diag_df):
            for column in frame:
                frame[column] = frame[column].astype(str)
                frame[column] = self._encode_class_labels(frame[column])

        for label in labels:
            self.dataset_no_useless = self.dataset_no_useless.drop(label, 1)
        self.dataset_no_useless = pd.concat([self.dataset_no_useless, main_diag_df, add_diag_df], axis=1, join='inner')
        self.dataset_no_useless = self.dataset_no_useless.fillna(-1)

    def _encode_class_labels(self, label):
        return self.encoder.fit_transform(label)

    def change_parsed_file(self, file_path):
        '''
            Change file with data. Isnt use now.
        :param file_path: new filepath
        :return: None
        '''
        self.to_parse = file_path

    @property
    def get_dataset_no_useless(self):
        '''
        :return: data after first Nan and incorrect replacing
        '''
        return self.dataset_no_useless

    @property
    def get_dataset_unmodified(self):
        '''
        :return: vanila data
        '''
        return self.dataset_unmodified

    @property
    def get_ages(self):
        '''
        :return: interval of ages
        '''
        return self.age_intervals
