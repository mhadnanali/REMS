from sklearn.utils import shuffle
import pandas as pd

class DataSetPartition():
    def __init__(self):
        print("Creating Training Dataset CSV Files")
        AllUsersDataset = self.DataSetLoad()
        self.DataPartition(AllUsersDataset)
    def DataSetLoad(self):
        AllUsersDataset = pd.read_csv("CSVFiles/IASC AllUsers.csv",index_col=None)
        print("Total dataset Shape: ",AllUsersDataset.shape)
        return AllUsersDataset

    def DataPartition(self, AllUsersDataset):
        print("Total dataset Shape: ",AllUsersDataset.shape)
        # Seperating Spam (deleted) and Genuine (not deleted) Users
        DeletedUsersDF = AllUsersDataset.loc[AllUsersDataset['y'] == 0] # 0 indicates that user is Spam
        NotDeleted = AllUsersDataset.loc[AllUsersDataset['y'] == 1]  # 1 indicates that user is Genuine
        print('\nDataframe Shape of Spam (Deleted) Users:\n', DeletedUsersDF.shape)
        print('\nDataframe Shape of Genuine (Not Deleted) Users:\n', NotDeleted.shape)
        NotDeleted = shuffle(NotDeleted, random_state=2001)
        selectedNotDeleted=NotDeleted.sample(n = 211394) # Taking sample random equal number of users as Spam users for training (50% both Spam and Genunie)
        print('\nDataframe Shape of Genuine (Not Deleted) After sample:\n', NotDeleted.shape)
        TrainingDataset = pd.concat([DeletedUsersDF,selectedNotDeleted], sort=False)
        print('\nDataframe Shape of Training Dataset:\n', TrainingDataset.shape)
        TrainingDataset.to_csv("CSVFiles/IASC TrainingData.csv", index=False)  #This Dataset (.csv) is already provided, uncomment this line to save the CSV file
        #return  TrainingDataset


    #TrainingDataset.head()