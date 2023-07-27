import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, jaccard_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class KNNTraining():
    def KNN(self,X_train, X_test, y_train, y_test):

        from sklearn.neighbors import KNeighborsClassifier
        Ks = 101
        mean_acc = np.zeros((Ks - 1))
        std_acc = np.zeros((Ks - 1))

        for n in range(1, Ks):
            start_time = time.time()
            # Train Model and Predict
            classifier = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
            y_pred_list = classifier.predict(X_test)
            mean_acc[n - 1] = accuracy_score(y_test, y_pred_list)

            std_acc[n - 1] = np.std(y_pred_list == y_test) / np.sqrt(y_pred_list.shape[0])
            X_train_Pred = classifier.predict(X_train)
            TrainingResult = accuracy_score(y_train, X_train_Pred)
            TestResult = accuracy_score(y_test, y_pred_list)
            F1_SCORE = f1_score(y_test, y_pred_list)  # , average='micro')
            ACCURACY = accuracy_score(y_test, y_pred_list)
            JaccardScore = jaccard_score(y_test, y_pred_list, pos_label=1)
            cfM = confusion_matrix(y_test, y_pred_list)
            RECALL = recall_score(y_test, y_pred_list)  # , average='micro')
            PRECISION = precision_score(y_test, y_pred_list)  # , average='micro')

            outputFile = "DataAnalysis/KNN_Analysis.csv"
            ex_time = time.time() - start_time
            if os.path.isfile(outputFile):
                resultsDF = pd.read_csv(outputFile, index_col=0)
            else:
                resultsDF = pd.DataFrame(
                    columns=['Algo', 'Kernal', 'depthORk', 'TrainSize', 'TestSize', 'NotDeleted', 'Deleted', 'Time',
                             'TrainingResult', 'TestResult',
                             'F1_SCORE', 'Accuracy', 'JaccardScore', 'PRECISION', 'RECALL', '0,0', '0,1', '1,0', '1,1'])
            values_to_add = {'Algo': 'KNN', 'Kernal': 'N/A', 'depthORk': n, 'TrainSize': X_train.shape[0],
                             'TestSize': X_test.shape[0],

                             'Time': ex_time, 'TrainingResult': TrainingResult, 'TestResult': TestResult,
                             'F1_SCORE': F1_SCORE, 'Accuracy': ACCURACY, 'JaccardScore': JaccardScore,
                             'PRECISION': PRECISION, 'RECALL': RECALL, '0,0': cfM[0][0], '0,1': cfM[0][1],
                             '1,0': cfM[1][0], '1,1': cfM[1][1]}
            row_to_add = pd.Series(values_to_add)
            resultsDF = resultsDF.append(row_to_add, ignore_index=True)
            resultsDF.to_csv(outputFile)
            print("Total Running time = {:.3f} seconds".format(ex_time), " of ", n)

        print(mean_acc)
        plt.plot(range(1, Ks), mean_acc, 'g')
        plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
        plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
        plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Neighbors (K)')
        plt.tight_layout()
        plt.savefig("Figures/Kvalue_KNN.pdf", format="pdf", dpi=1200)  # , bbox_inches="tight")

        from sklearn.neighbors import KNeighborsClassifier
        Ks = 91
        for i in range(20):
            # print ("Depth ", )

            for n in range(50, Ks):
                start_time = time.time()
                # Train Model and Predict
                classifier = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
                y_pred_list = classifier.predict(X_test)
                # mean_acc[n-1] = accuracy_score(y_test, y_pred_list)

                # std_acc[n-1]=np.std(y_pred_list==y_test)/np.sqrt(y_pred_list.shape[0])
                X_train_Pred = classifier.predict(X_train)
                TrainingResult = accuracy_score(y_train, X_train_Pred)
                TestResult = accuracy_score(y_test, y_pred_list)
                F1_SCORE = f1_score(y_test, y_pred_list)  # , average='micro')
                ACCURACY = accuracy_score(y_test, y_pred_list)
                JaccardScore = jaccard_score(y_test, y_pred_list, pos_label=1)
                cfM = confusion_matrix(y_test, y_pred_list)
                RECALL = recall_score(y_test, y_pred_list)  # , average='micro')
                PRECISION = precision_score(y_test, y_pred_list)  # , average='micro')

                outputFile = "DataAnalysis/KNN_Analysis_Fix_K.csv"
                ex_time = time.time() - start_time
                if os.path.isfile(outputFile):
                    resultsDF = pd.read_csv(outputFile, index_col=0)
                else:
                    resultsDF = pd.DataFrame(
                        columns=['Algo', 'Kernal', 'depthORk', 'TrainSize', 'TestSize', 'NotDeleted', 'Deleted', 'Time',
                                 'TrainingResult', 'TestResult',
                                 'F1_SCORE', 'Accuracy', 'JaccardScore', 'PRECISION', 'RECALL', '0,0', '0,1', '1,0',
                                 '1,1'])
                values_to_add = {'Algo': 'KNN', 'Kernal': 'N/A', 'depthORk': n, 'TrainSize': X_train.shape[0],
                                 'TestSize': X_test.shape[0],
                                 'Time': ex_time, 'TrainingResult': TrainingResult, 'TestResult': TestResult,
                                 'F1_SCORE': F1_SCORE, 'Accuracy': ACCURACY, 'JaccardScore': JaccardScore,
                                 'PRECISION': PRECISION, 'RECALL': RECALL, '0,0': cfM[0][0], '0,1': cfM[0][1],
                                 '1,0': cfM[1][0], '1,1': cfM[1][1]}
                row_to_add = pd.Series(values_to_add)
                resultsDF = resultsDF.append(row_to_add, ignore_index=True)
                resultsDF.to_csv(outputFile)
                print("Total Running time = {:.3f} seconds".format(ex_time), " of ", n)

