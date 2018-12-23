
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

# In[252]:


import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
from itertools import islice
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import pickle
import gzip
from PIL import Image
import os
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report


# ## Processing Input and Label Data

# In[253]:


clf1_m = []
clf2_m = []
clf3_m = []
clf4_m = []
clf1_u = []
clf2_u = []
clf3_u = []
clf4_u = []
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()


# In[254]:


USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []
for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)
            


# In[255]:


#creating one hot vector
TrainingTarget = np.zeros((training_data[1].shape[0], 10))
TrainingTarget[np.arange(training_data[1].shape[0]), training_data[1]] = 1
TrainingData = training_data[0]
#validation data
ValDataAct = np.zeros((validation_data[1].shape[0], 10))
ValDataAct[np.arange(validation_data[1].shape[0]), validation_data[1]] = 1
ValData = validation_data[0]
#Testing datas
TestDataAct = np.zeros((test_data[1].shape[0], 10))
TestDataAct[np.arange(test_data[1].shape[0]), test_data[1]] = 1
TestData = test_data[0]
#Combining validation and Training data
TrainingData = np.vstack((TrainingData,ValData))
TrainingTarget = np.vstack((TrainingTarget,ValDataAct))


# In[256]:


USPS_TestData = np.array(USPSMat)
USPS_test_target = np.array(USPSTar)
USPS_TestTarget = np.zeros((USPS_test_target.shape[0], 10))
USPS_TestTarget[np.arange(USPS_test_target.shape[0]), USPS_test_target] = 1


# In[257]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    print ("\nConfusion Matrix\n")
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ## Logistic Regression

# In[258]:


def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)): #computing the root mean squared error
        if(np.argmax(VAL_TEST_OUT[i]) == np.argmax(ValDataAct[i])): # classifying the regression output to three ranks 0,1,2 by rounding the y value to nearest even number
            #print(np.max(VAL_TEST_OUT[i]),np.max(ValDataAct[i]))
            counter = counter + 1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) #computes the ratio of correct prediction to total input
    return accuracy #accuracy 

def FindActivation(X, W):
#     print(X.shape)
#     print(W.shape)
    WX = np.dot(np.transpose(W),np.transpose(X)) #w=785*10 x=50000*785 WX=10*50000
    numerator = np.exp(WX) #10*50000 and avoid overfitting
    denom = np.sum(np.exp(WX), axis=0)#1*50000
    a = numerator/denom #softmax function #10*50000
#     print(np.shape(numerator),np.shape(denom),np.shape(a))
#     print(numerator.T[0],denom[0],a.T[0],np.sum(a.T[0]))
    return a

def LOGRModel(iter):    
    global TrainingTarget,TrainingData, ValDataAct, ValData , TestDataAct, TestData, USPS_TestData, USPS_TestTarget, clf1_m, clf1_u
    W_Now        = np.random.random((TrainingData.shape[1]+1,10))
    La           = 0.01
    learningRate = 0.1
    #AddBias
    X = np.ones((TrainingData.shape[0],1))
    TrainingData = np.hstack((TrainingData,X))
    X = np.ones((ValData.shape[0],1))
    ValData = np.hstack((ValData,X))
    X = np.ones((TestData.shape[0],1))
    TestData = np.hstack((TestData,X))
    X = np.ones((USPS_TestData.shape[0],1))
    USPS_TestData = np.hstack((USPS_TestData,X))
    print("After Adding Bias")
    print('----------TRAINING DATA--------------')
    print(TrainingTarget.shape)
    print(TrainingData.shape)
    print('---------VALIDATION DATA---------------')
    print(ValDataAct.shape)
    print(ValData.shape)
    print('----------MNIST TESTING DATA-------------')
    print(TestDataAct.shape)
    print(TestData.shape)
    print('---------- USPS TESTING DATA-------------')
    print(USPS_TestTarget.shape)
    print(USPS_TestData.shape)
    print('----------WEIGHT-------------')
    print(W_Now.shape)
    for i in range(0,iter): 
        for j in range(0, TrainingData.shape[0] - 500, 500):
            G = FindActivation(TrainingData[j:j+500], W_Now)
#             print(np.shape(G))                       
            val = np.subtract(G, np.transpose(TrainingTarget[j:j+500]))
#             print(np.shape(val)) 
            Delta_E_D = np.dot(val,TrainingData[j:j+500])/TrainingTarget[j:j+500].shape[0]
            La_Delta_E_W  = np.dot(La,W_Now) # Error regularization
            Delta_E       = np.add(np.transpose(Delta_E_D),La_Delta_E_W)  # adding regularization to gradient error
            Delta_W       = -np.dot(learningRate,Delta_E) # multipying learning rate to computed error
            W_T_Next      = W_Now + Delta_W # subtracting error from output
            W_Now         = W_T_Next # updating the weight
    #         print(W_Now)
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = FindActivation(TrainingData,W_T_Next) #10*10000
    acc_tr = GetErms(np.transpose(TR_TEST_OUT),TrainingTarget) #10000*10
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = FindActivation(ValData,W_T_Next) 
    acc_val = GetErms(np.transpose(VAL_TEST_OUT),ValDataAct)
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = np.transpose(FindActivation(TestData,W_T_Next))
    acc_test = GetErms(TEST_OUT,TestDataAct) 
    #-----------------USPS TestingData Accuracy---------------------#
    TEST_OUT_USPS      = np.transpose(FindActivation(USPS_TestData,W_T_Next))
    acc_test_usps = GetErms(TEST_OUT_USPS,USPS_TestTarget)
    print ('\n----------Gradient Descent Solution--------------------')
    print('learning rate %s' % learningRate)
    print('Lambda %s' % La)
    print ("Accuracy Training   = " + str(np.around(acc_tr,5)))
    print ("Accuracy Validation = " + str(np.around(acc_val,5)))
    print ("Accuracy Testing  MNIST  = " + str(np.around(acc_test,5)))
    print ("Accuracy Testing USPS   = " + str(np.around(acc_test_usps,5)))
    print(np.shape(np.argmax(TEST_OUT, axis = 1)))
    clf1_m = np.argmax(TEST_OUT, axis = 1)
    cm = confusion_matrix(test_data[1], np.argmax(TEST_OUT, axis = 1))
    plt.figure()
    plot_confusion_matrix(cm, classes=np.arange(10),
                          title='LR - Confusion matrix Mnist')
    plt.show()
    print(classification_report(test_data[1], np.argmax(TEST_OUT, axis = 1)))
    clf1_u = np.argmax(TEST_OUT_USPS, axis = 1)
    cm = confusion_matrix(USPS_test_target, np.argmax(TEST_OUT_USPS, axis = 1))
    plt.figure()
    plot_confusion_matrix(cm, classes=np.arange(10),
                          title='LR - Confusion matrix USPS')
    plt.show()
    print(classification_report(USPS_test_target, np.argmax(TEST_OUT_USPS, axis = 1)))


# ## Neural Networks

# In[259]:


def get_model():
    input_size = TrainingData.shape[1]
    drop_out = 0.3
    first_dense_layer_nodes  = 200
    final_dense_layer_nodes = 10
    second_dense_layer_nodes  = 100
    # Why do we need a model?
    # Why use Dense layer and then activation?
    # Why use sequential model with layers?
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # Why dropout?
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    
    model.add(Dropout(drop_out))
    
    model.add(Dense(final_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?
    
    model.summary()
    
    # Why use categorical_crossentropy?
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def TrainModel(iter):
    global clf2_m, clf2_u
    model = get_model()
    validation_data_split = 0.2
    num_epochs = iter
    model_batch_size = 211
    tb_batch_size = 32
    early_patience = 100

    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')


    # Process Dataset
    processedData= TrainingData
    print(processedData.shape)
    processedLabel = TrainingTarget
    print(processedLabel.shape)

    history = model.fit(processedData
                        , processedLabel
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                       )
    loss,accuracy = model.evaluate(TestData,TestDataAct)
    print("\n----------------NN CLASSIFIER------------------")
    print("\nMnist Results\n")
    print("loss,accuracy = " + str(loss) + " , " + str(accuracy * 100))
    y_pred_mnist = model.predict_classes(TestData)
    clf2_m = y_pred_mnist
    cm = confusion_matrix(test_data[1], y_pred_mnist)
    plt.figure()
    plot_confusion_matrix(cm, classes=np.arange(10),
                          title='NN - Confusion matrix Mnist')
    plt.show()
    print(classification_report(test_data[1], y_pred_mnist))
    print("\nUSPS Results\n")
    loss,accuracy = model.evaluate(USPS_TestData,USPS_TestTarget)
    print("loss,accuracy = " + str(loss) + " , " + str(accuracy * 100))
    y_pred_usps = clf2_u = model.predict_classes(USPS_TestData)
    cm = confusion_matrix(USPS_test_target, y_pred_usps)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=np.arange(10),
                          title='NN - Confusion matrix USPS')
    plt.show()
    print(classification_report(USPS_test_target, y_pred_usps))
    return history

def nn_main():
    history = TrainModel(20)
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))


# In[260]:


nn_main()


# # RandomForestClassifier

# In[261]:


from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators=10);
classifier2.fit(training_data[0], training_data[1])
y_pred = clf3_m = classifier2.predict(test_data[0])
y_pred_usps = clf3_u = classifier2.predict(USPS_TestData)
print("-----------------------------Random Forest Classifier----------------------------------------")

print("\nMnist Results\n")
print("Accuracy = " + str(classifier2.score(test_data[0],test_data[1]) * 100))
cm = confusion_matrix(test_data[1], y_pred )
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=np.arange(10),
                      title='RF - Confusion matrix Mnist')
plt.show()
print(classification_report(test_data[1], y_pred ))
print("\nUSPS Results\n")
print("Accuracy = " + str(classifier2.score(USPS_TestData,USPS_test_target) * 100))
cm = confusion_matrix(USPS_test_target, y_pred_usps)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=np.arange(10),
                      title='RF - Confusion matrix USPS')
plt.show()
print(classification_report(USPS_test_target, y_pred_usps))


# # SVM

# In[262]:


from sklearn.svm import SVC
X_train = training_data[0][0:10000]
Y_train = training_data[1][0:10000]
Mnist_test = test_data[0]
Usps_test = USPS_TestData
classifier1 = SVC(kernel='rbf', C=2, gamma = 0.05);
classifier1.fit(X_train, Y_train)
y_pred = clf4_m = classifier1.predict(Mnist_test)
y_pred_usps = clf4_u = classifier1.predict(Usps_test)
print("-------------------------------------SVM Classifier--------------------------------------------")
print("\nMnist Results\n")
print("Accuracy = " + str(classifier1.score(test_data[0],test_data[1]) * 100))
cm = confusion_matrix(test_data[1], y_pred )
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=np.arange(10),
                      title='SVM - Confusion matrix Mnist')
plt.show()
print(classification_report(test_data[1], y_pred ))
print("\nUSPS Results\n")
print("Accuracy = " + str(classifier1.score(USPS_TestData,USPS_test_target) * 100))
cm = confusion_matrix(USPS_test_target, y_pred_usps)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=np.arange(10),
                      title='SVM - Confusion matrix USPS')
plt.show()
print(classification_report(USPS_test_target, y_pred_usps))


# In[263]:

print("\n----------------LOGISTIC CLASSIFIER------------------")
LOGRModel(100)


# In[264]:


print("\n----------------ENSEMBLE CLASSIFIER------------------")
clf = np.vstack((clf1_m, clf2_m, clf3_m, clf4_m)).T
counter = 0
for i, arr in enumerate(clf):
    pred = np.argmax(np.bincount(arr))
    if(pred == test_data[1][i]):
        counter += 1
print("\nMNIST Accuracy = "+ str(float((counter*100))/clf.shape[0]))

clf = np.vstack((clf1_u, clf2_u, clf3_u, clf4_u)).T
counter = 0
for i, arr in enumerate(clf):
    pred = np.argmax(np.bincount(arr))
    if(pred == USPS_test_target[i]):
        counter += 1

print("\nUSPS Accuracy = "+ str(float((counter*100))/clf.shape[0]))

