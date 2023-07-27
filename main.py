from DTTraining import DTTraining
from DataPartition import DataSetPartition
from KNNTraining import KNNTraining
from MLPTraining import MLPTraining
from DataLoading import DataLoading
from NetworkXToGraph import DatasetCreation
from RFTraining import RFTraining
from SVMTraining import SVMTraining
from XGBTraining import XGBTraining


class REMS():
    def __init__(self):
        print("Starting")
        ###Uncomment these to create dataset from Graph
        createCSV = DatasetCreation()
        createCSV.GraphtoCSV(name='CSVFiles/AllUsersGraph.gpickle', Normalize=False)  # , Weights=True)
        DataSetPartition()  # Ca


if __name__ == '__main__':
    obj = REMS()
    dataLoad=DataLoading()
    X_train, X_test, y_train, y_test = dataLoad.LoadData()
    DTT= DTTraining()
    DTT.DecisionTrees(X_train, X_test, y_train, y_test)
    RFT = RFTraining()
    RFT.RandomForest(X_train, X_test, y_train, y_test)
    XGBT=XGBTraining()
    XGBT.XGB(X_train, X_test, y_train, y_test)
    KNNT=KNNTraining()
    KNNT.KNN(X_train, X_test, y_train, y_test)
    MLPT=MLPTraining()
    MLPT.MLP(X_train, X_test, y_train, y_test)
    SVMT=SVMTraining()
    SVMT.SVM(X_train, X_test, y_train, y_test)



