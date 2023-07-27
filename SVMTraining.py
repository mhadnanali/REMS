
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, jaccard_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import time

class SVMTraining():


    def SVM(self,X_train, X_test, y_train, y_test):
        from sklearn import svm

        kernals = ['sigmoid', 'poly', 'linear', 'rbf']  # ,  'poly','linear', 'rbf'
        for kernal in kernals:
            print('Executing: ', kernal)
            start_time = time.time()
            KERNAL = kernal
            clf = svm.SVC(kernel=KERNAL)
            clf.fit(X_train, y_train)
            y_pred_list = clf.predict(X_test)
            X_train_Pred = clf.predict(X_train)
            TrainingResult = accuracy_score(y_train, X_train_Pred)
            TestResult = accuracy_score(y_test, y_pred_list)
            F1_SCORE = f1_score(y_test, y_pred_list, average='micro')
            ACCURACY = accuracy_score(y_test, y_pred_list)
            JaccardScore = jaccard_score(y_test, y_pred_list, pos_label=1)
            cfM = confusion_matrix(y_test, y_pred_list)
            RECALL = recall_score(y_test, y_pred_list, average='micro')
            PRECISION = precision_score(y_test, y_pred_list, average='micro')
            print('Results of : ', kernal)
            print('F1_SCORE: ', F1_SCORE, ' ACCURACY :', ACCURACY, ' JaccardScore :', JaccardScore, ' RECALL :', RECALL,
                  ' PRECISION :', PRECISION)
            ax = sns.heatmap(cfM, annot=False, cmap='Blues')
            ax.set_title('SVM Results with ' + kernal);
            ax.set_xlabel('\nPredicted User Category')
            ax.set_ylabel('Actual User Category ');
            ex_time = time.time() - start_time
            #plt.show()
            print("Total Running time = {:.3f} seconds".format(ex_time), " of ", kernal)

