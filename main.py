import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wget
import os

# Let's import the processed data

path = os.getcwd()

#wget.download('https://raw.githubusercontent.com/ChristiaanvA/ml4pha_datasets/main/Skyserver_SQL_labels.csv',path)
#wget.download('https://raw.githubusercontent.com/ChristiaanvA/ml4pha_datasets/main/Skyserver_SQL_features.csv',path)

T = pd.read_csv(path+'/data/Skyserver_SQL_labels.csv').to_numpy().flatten()
X = pd.read_csv(path+'/data/Skyserver_SQL_features.csv').to_numpy()

# we need a subset of the data to impartially evaluate the performance of our classifier
# we shuffle X and T arrays, in such a way that each row still corresponds to each row! We split our target array T 
# into two arrays: T_train and T_test and corresponding two X_train and X_test arrays, where the train test contains 90% of 
# the datapoints and the test dataset contains 10% of the datapoints.

shuffler = np.random.permutation(len(T))
T_shuffled = T[shuffler]
X_shuffled = X[shuffler]
T_train, T_test = np.split(T_shuffled,[9000])
X_train, X_test = np.split(X_shuffled,[9000])

# We convert the train and test arrays to torch tensors.

X_train_tensor = torch.from_numpy(X_train).to(torch.float)
X_test_tensor = torch.from_numpy(X_test).to(torch.float)

T_train_tensor = torch.from_numpy(T_train).to(torch.long)
T_test_tensor = torch.from_numpy(T_test).to(torch.long)

# We create a Logistic Regression Pytorch model for multiclass classification with more than two 
# input parameters. 
class LogisticRegressionMultiClass(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionMultiClass, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self,x):
        y = torch.nn.functional.softmax(self.linear(x),dim=1)
        return y

# We train the code with the cross entropy loss, using the SGD optimizer. W use 15000 steps
# and a learning rate of 0.02 as a starting point.


model = LogisticRegressionMultiClass(np.shape(X_train_tensor)[1],len(np.unique(T_test_tensor)))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

model.train()

for epoch in range(15000):
    optimizer.zero_grad()

    # forward pass
    Y_pred = model(X_train_tensor)

    # compute loss
    loss = criterion(Y_pred, T_train_tensor)
    if epoch% 1000 == 0:
        print(loss.item())
    # backward pass
    loss.backward()
    optimizer.step()

# Although we aim to minimize the loss function, it hardly tells us as humans how well the classification scheme is performing.
# To get a better understanding of performance, we look at the accuracy, or the fraction of correct predictions from total predictions. 

model.eval()
T_test_pred = model(X_test_tensor)

T_test_pred = np.argmax(T_test_pred.detach().numpy(), axis=1)
T_test_true = T_test_tensor.detach().numpy()

data = dict = {
    'T_test_pred': T_test_pred,
    'T_test_true' : T_test_true
}

df_compare = pd.DataFrame(data)
df_compare['accuracy'] = df_compare.apply(lambda x: True if x['T_test_pred'] == x['T_test_true'] else False, axis=1)    
accuracy = df_compare['accuracy'].value_counts()[1]/len(df_compare['accuracy']) 

print('The model accuracy is {:.2f}%'.format(accuracy*100))         # we print the accuracy of the model

plt.figure(figsize=(16,10))
sns.countplot(x="accuracy", data=df_compare).set(title='Model Predictions')
plt.show()

'''
Analysis:
We can clearly see that the model predictions are good enough based on the accuracy, but we can also see how good is the model to predict individual classes.
'''

plt.figure(figsize=(16,10))
sns.countplot(x="accuracy", hue="T_test_pred", data=df_compare).set(title='Model Predictions in each Category')
plt.show()

'''
Analysis:
The above plot implies that the model predicts with very high accuracy quasars, while it can still predict good enough stars and galaxies.
'''