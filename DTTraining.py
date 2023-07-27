from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, jaccard_score, f1_score
import time
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
class DTTraining():


    def DecisionTrees(self,X_train, X_test, y_train, y_test ):
        for d in range(1, 21):
            print("Depth ", d)
            for i in range(20):
                CRITERIONs = ['gini', 'entropy', 'log_loss']
                for CRITERION in CRITERIONs:
                    start_time = time.time()
                    MAX_DEPTH = d

                    if CRITERION == 'log_loss':
                        MAX_DEPTH = d
                    # print(i,MAX_DEPTH)
                    classifier = DecisionTreeClassifier(criterion=CRITERION,
                                                        max_depth=MAX_DEPTH)  # {“gini”, “entropy”, “log_loss”},
                    classifier.fit(X_train, y_train)
                    y_pred_list = classifier.predict(X_test)
                    X_train_Pred = classifier.predict(X_train)
                    TrainingResult = accuracy_score(y_train, X_train_Pred)
                    TestResult = accuracy_score(y_test, y_pred_list)
                    F1_SCORE = f1_score(y_test, y_pred_list)  # , average='micro')
                    ACCURACY = accuracy_score(y_test, y_pred_list)
                    JaccardScore = jaccard_score(y_test, y_pred_list, pos_label=1)
                    cfM = confusion_matrix(y_test, y_pred_list)
                    RECALL = recall_score(y_test, y_pred_list)  # , average='micro')
                    PRECISION = precision_score(y_test, y_pred_list)  # , average='micro')
                    print('F1_SCORE: ',F1_SCORE,' ACCURACY :',ACCURACY,' JaccardScore :',JaccardScore, ' RECALL :',RECALL, ' PRECISION :',PRECISION)
                    outputFile = "DataAnalysis/DecisionTree_Analysis.csv"
                    ex_time = time.time() - start_time
                    if os.path.isfile(outputFile):
                        resultsDF = pd.read_csv(outputFile, index_col=0)
                    else:
                        resultsDF = pd.DataFrame(
                            columns=['Algo', 'Kernal', 'depthORk', 'TrainSize', 'TestSize', 'NotDeleted', 'Deleted',
                                     'Time', 'TrainingResult', 'TestResult',
                                     'F1_SCORE', 'Accuracy', 'JaccardScore', 'PRECISION', 'RECALL', '0,0', '0,1', '1,0',
                                     '1,1'])
                    values_to_add = {'Algo': 'DecisionTree', 'Kernal': CRITERION, 'depthORk': MAX_DEPTH,
                                     'TrainSize': X_train.shape[0], 'TestSize': X_test.shape[0],
                                     'Time': ex_time, 'TrainingResult': TrainingResult, 'TestResult': TestResult,
                                     'F1_SCORE': F1_SCORE, 'Accuracy': ACCURACY, 'JaccardScore': JaccardScore,
                                     'PRECISION': PRECISION, 'RECALL': RECALL, '0,0': cfM[0][0], '0,1': cfM[0][1],
                                     '1,0': cfM[1][0], '1,1': cfM[1][1]}
                    row_to_add = pd.Series(values_to_add)
                    resultsDF = resultsDF.append(row_to_add, ignore_index=True)
                    resultsDF.to_csv(outputFile)
                    print("Total Running time = {:.3f} seconds".format(ex_time), " of ", i, MAX_DEPTH)