import pandas as pd
import numpy as np

Data = pd.read_csv('titanic.csv')
del Data['Name']
Data['Sex'].replace({'male': 0, 'female':1}, inplace=True)
Survivedpd = Data['Survived']
del Data['Survived']
Datanp = pd.DataFrame(Data).to_numpy()
Survivednp = pd.DataFrame(Survivedpd).to_numpy().T

bias = np.ones((887, 1))
Datanp = np.append(bias, Datanp, axis=1).T
training_size = 887
print('TRAINING DATA')
print(Datanp.shape)
print(Datanp)
print('ANSWER DATA')
print(Survivednp.shape)
print(Survivednp)