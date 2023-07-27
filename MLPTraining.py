import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, jaccard_score, f1_score
import pandas as pd
from sklearn.metrics import average_precision_score

class MLPTraining():


    def MLP(self,X_train, X_test, y_train, y_test):
        EPOCHS = 1000
        BATCH_SIZE = 338230
        LEARNING_RATE = 0.01
        self.epoch_test=50
        train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train.to_numpy()))
        test_data = TestData(torch.FloatTensor(X_test))
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(dataset=test_data, batch_size=1)
        self.y_test=y_test
        self.device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        inputdim = X_train.shape
        print(inputdim)
        model = BinaryClassification(int(inputdim[1]))
        model.to(self.device)
        print(model)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.X_test=X_test
        self.X_train=X_train
        self.training(EPOCHS, model,train_loader,test_loader,criterion,optimizer)
        self.testing(test_loader)

    def training(self,EPOCHS,model,train_loader,test_loader,criterion,optimizer):
        from tqdm.notebook import tqdm
        for i in range(1):
            print("This is times:", i)
            results = (0, 0)
            model.train()
            t = 1
            F1_SCORE = 0
            ACCURACY = 0
            RECALL = 0
            PRECISION = 0
            with tqdm(total=EPOCHS, desc='(T)') as pbar:
                print(t + 1)
                for e in range(1, EPOCHS + 1):
                    #    print(e)
                    epoch_loss = 0
                    epoch_acc = 0
                    fl = 0
                    pbar.update()
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        optimizer.zero_grad()

                        y_pred = model(X_batch)
                        # print(e,y_pred.shape)
                        loss = criterion(y_pred, y_batch.unsqueeze(1))
                        acc = self.binary_acc(y_pred, y_batch.unsqueeze(1))

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        epoch_acc += acc.item()
                        if fl == 0:
                            savefile = y_pred
                            y_orignal = y_batch.unsqueeze(1)
                            X_orignal = X_batch
                            fl = 1
                        elif fl == 1:
                            savefile = torch.cat((savefile, y_pred))
                            y_orignal = torch.cat((y_orignal, y_batch.unsqueeze(1)))
                            X_orignal = torch.cat((X_orignal, X_batch))

                    pbar.set_postfix({'Acc': epoch_acc / len(train_loader)})
                    results = (e, epoch_acc / len(train_loader))
                    if e % 30 == 0:
                        print(
                            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

                    if epoch_acc / len(train_loader) > results[1] + 0.1:

                        # print(results)
                        print(
                            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
                        # xs = torch.sigmoid(savefile)

                        # x_df = pd.DataFrame(X_train_csv)
                        # x_df['Orignal']=y_train_csv

                        x_df = pd.DataFrame(X_orignal.cpu().detach().numpy())
                        x_df['Orignal'] = y_orignal.cpu().detach().numpy()
                        x_df['Prediction'] = savefile.cpu().detach().numpy()
                        if results[1] > 95:
                            x_df.to_csv(str(results[1]) + ' training results.csv')

                    if e % self.epoch_test == 0:
                        print(
                            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
                        y_pred_list = []
                        model.eval()
                        pred_prob = []
                        with torch.no_grad():
                            for X_batch in tqdm(test_loader):
                                X_batch = X_batch.to(self.device)
                                y_test_pred = model(X_batch)

                                y_test_pred = torch.sigmoid(y_test_pred)
                                pred_prob.append(y_test_pred.cpu().numpy()[0])
                                y_pred_tag = torch.round(y_test_pred)
                                y_pred_list.append(y_pred_tag)

                        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
                        # print (y_pred_list)
                        from sklearn.metrics import confusion_matrix
                        cfM = confusion_matrix(self.y_test, y_pred_list)
                        from sklearn.metrics import average_precision_score
                        average_precision = average_precision_score(self.y_test, y_pred_list)

                        t_F1_SCORE = f1_score(self.y_test, y_pred_list)  # , average='micro')
                        t_ACCURACY = accuracy_score(self.y_test, y_pred_list)
                        t_JaccardScore = jaccard_score(self.y_test, y_pred_list, pos_label=1)
                        t_RECALL = recall_score(self.y_test, y_pred_list)  # , average='micro')
                        t_PRECISION = precision_score(self.y_test, y_pred_list)  # , average='micro')

                        if F1_SCORE < t_F1_SCORE:
                            F1_SCORE = t_F1_SCORE
                            ACCURACY = t_ACCURACY
                            JaccardScore = t_JaccardScore
                            RECALL = t_RECALL
                            PRECISION = t_PRECISION
                            cf_matrix = confusion_matrix(self.y_test, y_pred_list)
                            path = "Model/ModelNotCleaned.pth"
                            torch.save(model, path)
                            print("Saved Model")
                            print(
                                f'Saved Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
                            print('F1_SCORE: ', F1_SCORE, ' ACCURACY :', ACCURACY, ' JaccardScore :', JaccardScore,
                                  ' RECALL :', RECALL, ' PRECISION :', PRECISION)

                    # print('Average precision-recall score: {0:0.2f}'.format(average_precision))

                    # print(confusion_matrix(y_test, y_pred_list))
                    # from sklearn.metrics import accuracy_score
                    # print(accuracy_score( y_test, y_pred_list))
                    # print(results)
                import os
                # F1_SCORE=f1_score(y_test, y_pred_list, zero_division=1)
                outputFile = "DataAnalysis/MLPResults.csv"
                if os.path.isfile(outputFile):
                    resultsDF = pd.read_csv(outputFile, index_col=0)
                else:
                    resultsDF = pd.DataFrame(
                        columns=['Total_Epoch', 'Train', 'Test', 'RECALL', 'JaccardScore', 'Epoch', 'Result', 'TestACC',
                                 'PRECISION', 'F1_SCORE', '0,0', '0,1', '1,0', '1,1'])

                values_to_add = {'Total_Epoch': EPOCHS, 'Train': self.X_train.shape[0], 'Test': self.X_test.shape[0],
                                 'RECALL': RECALL, 'JaccardScore': JaccardScore,
                                 'Epoch': results[0],
                                 'Result': results[1],
                                 'TestACC': ACCURACY,
                                 'PRECISION': PRECISION, 'F1_SCORE': F1_SCORE, '0,0': cfM[0][0], '0,1': cfM[0][1],
                                 '1,0': cfM[1][0], '1,1': cfM[1][1]}
                row_to_add = pd.Series(values_to_add)
                resultsDF = resultsDF.append(row_to_add, ignore_index=True)
                resultsDF.to_csv(outputFile)
                from sklearn.metrics import confusion_matrix
                # accuraci= classification_report( y_test, y_pred_list)#, target_names=labels)
                # Get the confusion matrix

                print("Confusion Matrix: ", cf_matrix)
                print("F1 Score: ", F1_SCORE)
                import seaborn as sns
                import matplotlib.pyplot as plt
                ax = sns.heatmap(cf_matrix, annot=False, cmap='Blues')
                ax.set_title('Seaborn Confusion Matrix with labels\n\n');
                ax.set_xlabel('\nPredicted User Category')
                ax.set_ylabel('Actual User Category ');
                ## Ticket labels - List must be in alphabetical order
                # ax.xaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
                # ax.yaxis.set_ticklabels(['Setosa','Versicolor', 'Virginia'])
                ## Display the visualization of the Confusion Matrix.
                #plt.show()

    def binary_acc(self,y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()

        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def testing(self,test_loader):
        y_test=self.y_test
        path = "Model/ModelNotCleaned.pth"
        y_pred_list = []
        model = torch.load(path)
        # model.load_state_dict(torch.load(path))
        model.eval()
        pred_prob = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = model(X_batch)
                # y_test_pred_prob=y_test_pred
                y_test_pred = torch.sigmoid(y_test_pred)
                # pred_prob.append(y_test_pred.cpu().numpy()[0])
                pred_prob.append(y_test_pred.item())
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag)

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        # print (y_test_pred_prob)
        cfM = confusion_matrix(y_test, y_pred_list)
        average_precision = average_precision_score(y_test, y_pred_list)
        t_F1_SCORE = f1_score(y_test, y_pred_list)  # , average='micro')
        t_ACCURACY = accuracy_score(y_test, y_pred_list)
        t_JaccardScore = jaccard_score(y_test, y_pred_list, pos_label=1)
        t_RECALL = recall_score(y_test, y_pred_list)  # , average='micro')
        t_PRECISION = precision_score(y_test, y_pred_list)  # , average='micro')
        print("Acc: ", t_F1_SCORE, "Acc: ", t_ACCURACY)
## train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class BinaryClassification(nn.Module):
    def __init__(self, inputdim):
        super(BinaryClassification, self).__init__()
        # Number of input features .
        self.layer_1 = nn.Linear(inputdim, 900)  # 1000
        self.layer_2 = nn.Linear(900, 450)
        self.layer_3 = nn.Linear(450, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(900)
        self.batchnorm2 = nn.BatchNorm1d(450)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)

        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)

        x = self.dropout(x)
        x = self.layer_out(x)
        # x = self.relu(x)
        # x = torch.sigmoid(x)
        return x

