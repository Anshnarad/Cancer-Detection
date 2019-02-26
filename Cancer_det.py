import time                                             #Provides various time related functions.
import sys                                              #provides functions that interact with interpreter.
import pandas as pd                                     #high performance library for data structures.
import pylab as pl                                      #matplotlib based interface for plotting.
import numpy as np                                      #Math library enables to compute efficiently and effectively.
import scipy.optimize as opt                            #It is a solver for nonlinear problems eg curve fitting.
from sklearn import preprocessing                       #Provides several utility functions to transform a raw dataset to more suitable representation.
from sklearn.model_selection import train_test_split    #Split arrays and matrices to random train and test data set.
import matplotlib.pyplot as plt                         #library used for plotting graphs.
timer=2
cell_char = pd.read_csv("cell_samples.csv")               #Read through the csv file line by line.

cell_char.head()                                          #Return results for n rows.

ax = cell_char[cell_char['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='malignant'); #For plotting scatterpoints on the scatterplot.
cell_char[cell_char['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='lime', label='benign', ax=ax); #For plotting scatterpoints on the scatterplot.

cell_char.dtypes                                          #Returns data type of each column.

cell_char = cell_char[pd.to_numeric(cell_char['BareNuc'], errors='coerce').notnull()]  #take column and convert to numeric and coerce when specified. Does this only when value is not null.
cell_char['BareNuc'] = cell_char['BareNuc'].astype('int')   #Converts the column to int type.

cell_char.dtypes                                          #Returns data type of each column.

feature_df = cell_char[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']] #feature_dataframe is initialised here.

X = np.asarray(feature_df)                              #feature dataframe is stored as array inside X.


cell_char['Class'] = cell_char['Class'].astype('int')       #Converts column class to type int.
y = np.asarray(cell_char['Class'])                        #class column is stored as array as y.

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.22, random_state=4) #splitting training and testing data set with training being approximately 0.75% of total dataset.

print("\n")
gen = ' ********************* CANCER DETECTION MODEL *********************'
for i in gen:
    print (i, end='')
    sys.stdout.flush()
    time.sleep(0.05)

print("\n")

gen= ' *****************************************************************'
for i in gen:
    print(i, end='')
    sys.stdout.flush()
    time.sleep(0.05)

time.sleep(timer)
print("\n\n")
print ('Train set:', X_train.shape,  y_train.shape)     #printing number of training data sets used.
print ('Test set :', X_test.shape,  y_test.shape)       #printing number of testing data sets used.

print("\n")
time.sleep(1)
from sklearn import svm                                 #provides variety of kernel functions that uses subsets of training points for decision functions.

clf = svm.SVC(gamma='auto',kernel='rbf')                #Specifies kernel type to be used in support vector classification.
clf.fit(X_train, y_train)                               #training the model using fit method.
predi = clf.predict(X_test)                             #learns the link between training and testing data and returns the label for an unlabeled tests.

from sklearn.metrics import classification_report, confusion_matrix #for making classification report and generating confusion matrix.
import itertools                                        #Functions to create iterators for efficient looping.

print("\n")
def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Reds): #plots confusion matrix with required colormapping and title.


    print('Confusion matrix :-')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)   #plots confusion matrix.
    plt.title(title)                                     # gives title to it.
    plt.colorbar()                                       # Generates colormap bar.

    check_p = np.arange(len(classes))                    #for setting current location of tick.

    plt.xticks(check_p, classes)                         #plot tick on x axis with label.
    plt.yticks(check_p, classes)                         #plot tick on y axis with label.

    fmt = 'd'                                            #format type integer
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #looping over data dimensions and creating text annotations.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()                                   #automatic adjustment of figure.
    plt.ylabel('True label')                             #label at y axis.
    plt.xlabel('Predicted label')                        #label at x axis.

cnf_matrix = confusion_matrix(y_test, predi, labels=[2,4]) # Compute confusion matrix
np.set_printoptions(precision=2)

print (classification_report(y_test, predi))             #print classification report.


plt.figure()                                             # Plot non-normalized confusion matrix

plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix') #plot confusion matrix.
print("\n")
print('THE WEIGHTED MEAN OF PRECISION AND RECALL IS :-')

from sklearn.metrics import f1_score                     #import f1_score.
print(f1_score(y_test, predi, average='weighted'))       #print f1_score.
print("\n ********************************************************************\n")
gen= ' ********************************END*********************************'
for i in gen:
    print(i, end='')
    sys.stdout.flush()
    time.sleep(0.05)
plt.show()                                               #output the graphs.
