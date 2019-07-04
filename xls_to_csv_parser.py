import pandas as pd


def parser():
    data_xls = pd.read_excel('datasetxls.xlsx', 'qword-20181205105452814', index_col=0)
    data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')

    csv_dataset = pd.read_csv('res_dataset.csv', sep=';')
    csv_dataset = csv_dataset.drop('№', 1)
    csv_dataset = csv_dataset.drop('%Код экземпляра', 1)

    for i in range(len(csv_dataset['Пол'].values)):
        if csv_dataset['Пол'][i] == 'Мужской':
            csv_dataset['Пол'].values[i] = 1
        else:
            csv_dataset['Пол'].values[i] = 2

    csv_dataset = csv_dataset.fillna(-1)
    print(csv_dataset)
    return csv_dataset

parser()