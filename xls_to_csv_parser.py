import pandas as pd
from sklearn import preprocessing

def parser():
    data_xls = pd.read_excel('datasetxls.xlsx', 'qword-20181205105452814', index_col=0)
    data_xls.to_csv('res_dataset.csv', encoding='utf-8', sep=';')
    le = preprocessing.LabelEncoder()

    csv_dataset = pd.read_csv('res_dataset.csv', sep=';')
    csv_dataset = csv_dataset.drop('№', 1)
    csv_dataset = csv_dataset.drop('%Код экземпляра', 1)
    tr = csv_dataset['Пол'].values
    #tr = le.fit_transform(tr)
    for i in tr:
        print(i)
    #csv_dataset['Пол'].values =
    #csv_dataset['Пол'] = csv_dataset['Пол'].replace('Мужской', 0)
    #csv_dataset['Пол'] = csv_dataset['Пол'].replace('Женский', 1)



    csv_dataset = csv_dataset.fillna(-1)
    print(csv_dataset)
    return csv_dataset

parser()