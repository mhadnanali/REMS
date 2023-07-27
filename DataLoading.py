import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class DataLoading():
    def __init__(self):
        print("Creating Training Dataset CSV Files")


    def LoadData(self):
        TrainingDataset = pd.read_csv("CSVFiles/IASC TrainingData.csv")
        sns.countplot(x = 'y', data=TrainingDataset)
        print('\nDataframe Shape of Total Training Dataset:\n', TrainingDataset.shape)
        datasetShuf = shuffle(TrainingDataset, random_state=98567)
        X = datasetShuf.iloc[:, 0:-1]
        y = datasetShuf.iloc[:, -1]
        #print (datasetShuf.shape)
        #print (X.shape)
        #print (y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10234) # 80% training and 20% testing data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #print (type(X))
        #print (type(y))
        print ('Train Dataset Shape:\n', (X_train.shape))
        print ('Test Dataset Shape:\n',X_test.shape)
        #print('Sample X_train', X_train[0:1])
        #print('Sample X_test',X_test[0:1])
        return X_train, X_test, y_train, y_test