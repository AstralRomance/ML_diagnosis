import pandas as pd



class Parser:
    def parser(self):
        data_xls = pd.read_excel('datasetxls.xlsx', 'qword-20181205105452814', index_col=0)
        data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')

        output_dataset = pd.read_csv('res_dataset.csv', sep=';')
        output_dataset = output_dataset.drop('№', 1)
        output_dataset = output_dataset.drop('%Код экземпляра', 1)

        for i in range(len(output_dataset['Пол'].values)):
            if output_dataset['Пол'][i] == 'Мужской':
                output_dataset['Пол'].values[i] = 1
            else:
                output_dataset['Пол'].values[i] = 2

        output_dataset = output_dataset.fillna(-1)
        #print(output_dataset)
        return output_dataset

#parser()