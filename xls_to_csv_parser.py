import pandas as pd



class Parser:
    def parser(self):
        data_xls = pd.read_excel('datasetxls.xlsx', 'qword-20181205105452814', index_col=0)
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

        for i, j in enumerate(dataset['Возраст'].values):
            if j < 0.0:
                dataset['Возраст'].values[i] = int(-1)

            if j < 17.0 and j > 0.0:
                dataset['Возраст'].values[i] = int(0)

            if j >= 17.0 and j < 21.0:
                dataset['Возраст'].values[i] = int(1)

            if j >= 21.0 and j < 55.0:
                dataset['Возраст'].values[i] = int(2)

            if j >= 55.0 and j < 75.0:
                dataset['Возраст'].values[i] = int(3)

            if j >= 75.0 and j < 90.0:
                dataset['Возраст'].values[i] = int(4)

            if j >= 90.0:
                dataset['Возраст'].values[i] = int(5)

        with open('temp2.txt', 'w') as temp:
            for i in dataset['Пол'].values:
                temp.write(str(i))

'''
        for i, j in enumerate(dataset['Возраст'].values):
            #print(dataset['Возраст'].values[j])
            print(dataset['Возраст'].values[i])
'''
prs = Parser()
prs.parser()
