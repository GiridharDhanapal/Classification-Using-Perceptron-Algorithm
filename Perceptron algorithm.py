import numpy as np
import pandas as pd
colnames = ['First','Second','Third','Forth','Class'] # Assigning column names for the data
df = pd.read_csv('train.data',names=colnames ,header=None) # Reading the csv file using pandas
df1 = pd.read_csv('test.data',names=colnames ,header=None)

class Perceptron:
  def train(self, X, Y, iterations):
    weights = np.zeros(X.shape[1])
    bias = 0
    for i in range(iterations):
      for count, value in enumerate(X):
        activation = np.dot(weights, value) + bias
        if Y[count] * activation <= 0:
          weights = weights + Y[count] * value
          bias = bias + Y[count]
    return weights, bias

  def test(self, X, weights, bias):
    activation = np.dot(X, weights) + bias
    return np.sign(activation)


class MC_Perceptron:
  def train(self, X, Y, iterations, lam):  # lam is the regularization coefficient
    weights = np.array(np.zeros(X.shape[1]))
    bias = 0
    for i in range(iterations):
      for j, k in enumerate(X):
        activation = np.dot(weights, k) + bias
        if Y[j] * activation <= 0:
          weights = (1 - 2 * lam) * weights + Y[j] * k
          bias = bias + Y[j]
        else:
          weights = (1 - 2 * lam) * weights
          bias = bias
    return weights, bias

  def test(self, X, weights, bias):
    activation = np.dot(X, weights) + bias
    return np.sign(activation)
 
# Training for class 1 and class 2
c12 = df[(df['Class'] == 'class-1') | (df['Class'] == 'class-2')]
c12.loc[c12['Class'] == 'class-1', 'Class'] = 1
c12.loc[c12['Class'] != 1, 'Class'] = -1
X = c12[c12.columns[0:4]].values
Y = c12[c12.columns[4]].values
weights,bias = Perceptron.train(X,Y,20)

# Testing for class 1 and class 2
c12Test = df1[(df1['Class'] == 'class-1') | (df1['Class'] == 'class-2')]
c12Test.loc[c12Test['Class'] == 'class-1', 'Class'] = 1
c12Test.loc[c12Test['Class'] != 1, 'Class'] = -1
XTest = c12Test[c12Test.columns[0:4]].values
YTest = c12Test[c12Test.columns[4]].values
Out = sum(Y == Perceptron.test(X, weights, bias))
Acc_Train = Out*100/len(Y)
Output = sum(YTest == Perceptron.test(XTest,weights,bias))
Acc_Test = Output * 100 / len(YTest)
print("Accuracy for class 1 vs class 2 for Train Data and Test Data is {} & {}".format(Acc_Train, Acc_Test))

# Training for class 2 and class 3
c23 = df[(df['Class'] == 'class-2') | (df['Class'] == 'class-3')]
c23.loc[c23['Class'] == 'class-2', 'Class'] = 1
c23.loc[c23['Class'] != 1, 'Class'] = -1
X = c23[c23.columns[0:4]].values
Y = c23[c23.columns[4]].values
weights,bias = Perceptron.train(X,Y,20)

# Testing for class 2 and class 3
c23Test = df1[(df1['Class'] == 'class-2') | (df1['Class'] == 'class-3')]
c23Test.loc[c23Test['Class'] == 'class-2', 'Class'] = 1
c23Test.loc[c23Test['Class'] != 1, 'Class'] = -1
XTest = c23Test[c23Test.columns[0:4]].values
YTest = c23Test[c23Test.columns[4]].values
Out = sum(Y == Perceptron.test(X, weights, bias))
Acc_Train = Out*100/len(Y)
Output = sum(YTest == Perceptron.test(XTest,weights,bias))
Acc_Test = Output * 100 / len(YTest)
print("Accuracy for class 2 vs class 3 for Train Data and Test Data is {} & {}".format(Acc_Train, Acc_Test))

# Training for class 1 and class 3
c13 = df[(df['Class'] == 'class-1') | (df['Class'] == 'class-3')]
c13.loc[c13['Class'] == 'class-1', 'Class'] = 1
c13.loc[c13['Class'] != 1, 'Class'] = -1
X = c13[c13.columns[0:4]].values
Y = c13[c13.columns[4]].values
weights,bias = Perceptron.train(X,Y,20)

# Testing for class 1 and class 3
c13Test = df1[(df1['Class'] == 'class-1') | (df1['Class'] == 'class-3')]
c13Test.loc[c13Test['Class'] == 'class-1', 'Class'] = 1
c13Test.loc[c13Test['Class'] != 1, 'Class'] = -1
XTest = c13Test[c13Test.columns[0:4]].values
YTest = c13Test[c13Test.columns[4]].values
Out = sum(Y == Perceptron.test(X, weights, bias))
Acc_Train = Out*100/len(Y)
Output = sum(YTest == Perceptron.test(XTest,weights,bias))
Acc_Test = (round(Output * 100 / len(YTest),2))
print("Accuracy for class 1 vs class 3 for Train Data and Test Data is {} & {}".format(Acc_Train, Acc_Test))

# Class 1 vs Rest

# Train
colnames = ['First','Second','Third','Forth','Class']
df = pd.read_csv('train.data',names=colnames ,header=None)
df['Class'] = np.where(df['Class'] == 'class-1',1,-1)
X = df[df.columns[0:4]].values
Y = df[df.columns[4]].values
weights,bias = Perceptron.train(X,Y,20)
# Test
colnames = ['First','Second','Third','Forth','Class']
df1 = pd.read_csv('test.data',names=colnames ,header=None)
df1['Class'] = np.where(df1['Class'] == 'class-1',1,-1)
XTest = df1[df1.columns[0:4]].values
YTest = df1[df1.columns[4]].values
Out = sum(Y == Perceptron.test(X, weights, bias))
Acc_Train = Out*100/len(Y)
Output = sum(YTest == Perceptron.test(XTest, weights, bias))
Acc_Test = round(Output*100/len(YTest),2)
print("Accuracy for class 1 vs Rest for Train Data and Test Data is {} & {}".format(Acc_Train, Acc_Test))

# Class 2 vs Rest

# Train
colnames = ['First','Second','Third','Forth','Class']
df = pd.read_csv('train.data',names=colnames ,header=None)
df['Class'] = np.where(df['Class'] == 'class-2',1,-1)
X = df[df.columns[0:4]].values
Y = df[df.columns[4]].values
weights,bias = Perceptron.train(X,Y,20)
# Test
colnames = ['First','Second','Third','Forth','Class']
df1 = pd.read_csv('test.data',names=colnames ,header=None)
df1['Class'] = np.where(df1['Class'] == 'class-2',1,-1)
XTest = df1[df1.columns[0:4]].values
YTest = df1[df1.columns[4]].values
Out = sum(Y == Perceptron.test(X, weights, bias))
Acc_Train = Out*100/len(Y)
YTestPredicted = Perceptron.test(XTest, weights, bias)
Output = sum(YTest == YTestPredicted)
Acc_Test = Output*100/len(YTest)
print("Accuracy for class 2 vs Rest for Train Data and Test Data is {} & {}".format(Acc_Train, Acc_Test))

# class 3 vs Rest

# Train
colnames = ['First','Second','Third','Forth','Class']
df = pd.read_csv('train.data',names=colnames ,header=None)
df['Class'] = np.where(df['Class'] == 'class-3',1,-1)
X = df[df.columns[0:4]].values
Y = df[df.columns[4]].values
weights,bias = Perceptron.train(X,Y,20)
# Test
colnames = ['First','Second','Third','Forth','Class']
df1 = pd.read_csv('test.data',names=colnames ,header=None)
df1['Class'] = np.where(df1['Class'] == 'class-3',1,-1)
XTest = df1[df1.columns[0:4]].values
YTest = df1[df1.columns[4]].values
Out = sum(Y == Perceptron.test(X, weights, bias))
Acc_Train = Out*100/len(Y)
Output = sum(YTest == Perceptron.test(XTest, weights, bias))
Acc_Test = Output*100/len(YTest)
print("Accuracy for class 3 vs Rest for Train Data and Test Data is {} & {}".format(Acc_Train, Acc_Test))

# Using Regularisation Coefficient

# Class 1 vs Other
colnames = ['First','Second','Third','Forth','Class']
df = pd.read_csv('train.data',names=colnames ,header=None)
df['Class'] = np.where(df['Class'] == 'class-1',1,-1)
X = df[df.columns[0:4]].values
Y = df[df.columns[4]].values
df1 = pd.read_csv('test.data',names=colnames ,header=None)
df1['Class'] = np.where(df1['Class'] == 'class-1',1,-1)
XTest = df1[df1.columns[0:4]].values
YTest = df1[df1.columns[4]].values
for x in [0.01, 0.1, 1.0, 10.0, 100.0]:
 weights,bias = MC_Perceptron.train(X,Y,20,x)
 Out = sum(Y == MC_Perceptron.test(X,weights,bias))
 Acc_Train = Out*100/len(Y)
 YTestPredicted = MC_Perceptron.test(XTest, weights, bias)
 Output = sum(YTest == YTestPredicted)
 Acc_Test = (Output*100/len(YTest))
 print("Accuracy of 1 vs Rest by using Reg_Coef as",x,"for Train and Test data is {} & {}".format(Acc_Train, Acc_Test))

# Class 2 vs Other
colnames = ['First','Second','Third','Forth','Class']
df = pd.read_csv('train.data',names=colnames ,header=None)
df['Class'] = np.where(df['Class'] == 'class-2',1,-1)
X = df[df.columns[0:4]].values
Y = df[df.columns[4]].values
df1 = pd.read_csv('test.data',names=colnames ,header=None)
df1['Class'] = np.where(df1['Class'] == 'class-2',1,-1)
XTest = df1[df1.columns[0:4]].values
YTest = df1[df1.columns[4]].values
for x in [0.01, 0.1, 1.0, 10.0, 100.0]:
 weights,bias = MC_Perceptron.train(X,Y,20,x)
 Out = sum(Y == MC_Perceptron.test(X,weights,bias))
 Acc_Train = Out*100/len(Y)
 Output = sum(YTest == MC_Perceptron.test(XTest, weights, bias))
 Acc_Test = (Output*100/len(YTest))
 print("Accuracy of 2 vs Rest by using Reg_Coef as",x,"for Train and Test data is {} & {}".format(Acc_Train, Acc_Test))

# Class 3 vs Other 
colnames = ['First','Second','Third','Forth','Class']
df = pd.read_csv('train.data',names=colnames ,header=None)
df['Class'] = np.where(df['Class'] == 'class-3',1,-1)
X = df[df.columns[0:4]].values
Y = df[df.columns[4]].values
df1 = pd.read_csv('test.data',names=colnames ,header=None)
df1['Class'] = np.where(df1['Class'] == 'class-3',1,-1)
XTest = df1[df1.columns[0:4]].values
YTest = df1[df1.columns[4]].values
for x in [0.01, 0.1, 1.0, 10.0, 100.0]:
 weights,bias = MC_Perceptron.train(X,Y,20,x)
 Out = sum(Y == MC_Perceptron.test(X,weights,bias))
 Acc_Train = Out*100/len(Y)
 YTestPredicted = MC_Perceptron.test(XTest, weights, bias)
 Output = sum(YTest == YTestPredicted)
 Acc_Test = (Output*100/len(YTest))
 print("Accuracy of 3 vs Rest by using Reg_Coef as",x,"for Train and Test data is {} & {}".format(Acc_Train, Acc_Test))