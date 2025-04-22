#AF=0
#Brady=1
#Tachy=2
#Normal=3
from sklearn.metrics import accuracy_score
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

labels=[]
features=[]
with open('Features.csv','r') as aba:
    data=csv.reader(aba)
    
    for i in data:
        a=i[:-1]
        #print(a)
        features.append(a)
        
        labels.append(i[-1])
x=features[1:]

y=labels[1:]

x = np.asarray(x)

y=np.asarray(y)

x=x.astype(np.float64).tolist()
y=y.astype(np.float64).tolist()


#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


print('Decision Tree Classifier:',accuracy_score(y_test,y_pred))

'''
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("dtcfinal.png")'''


#-------------------------------------------------------------------------labels=[]
svmlabels=[]
svmfeatures=[]
with open('Features.csv','r') as svmaba:
    svmdata=csv.reader(svmaba)
    
    for i in svmdata:
        a=i[:-2]
        #print(a)
        svmfeatures.append(a)
        
        svmlabels.append(i[-1])
svm_x=svmfeatures[1:]

svm_y=svmlabels[1:]

svm_x = np.asarray(svm_x)
svm_y=np.asarray(svm_y)

svm_x=svm_x.astype(np.float64).tolist()
svm_y=svm_y.astype(np.float64).tolist()


X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(svm_x, svm_y, test_size=0.3, random_state=1)





#----------------------------------------------------------------------------------

#SVM-1

svm_model = svm.SVC(gamma='scale',kernel='linear',C=1.0)

svm_model.fit(X_train_svm,y_train_svm) 

predict_y_svm=svm_model.predict(X_test_svm)

print('SVM linear:',accuracy_score(y_test_svm,predict_y_svm))

#SVM-2

svm_model2 = svm.SVC(gamma='scale',kernel='rbf',C=1.0)

svm_model2.fit(X_train_svm,y_train_svm) 

predict_y_rbf=svm_model2.predict(X_test_svm)

print('SVM rbf:',accuracy_score(y_test_svm,predict_y_rbf))



#SVM-3


svm_model3= svm.SVC(gamma='scale',kernel='poly', degree=3, C=1.0)

svm_model3.fit(X_train_svm,y_train_svm) 

predict_y_poly=svm_model3.predict(X_test_svm)

print('SVM poly:',accuracy_score(y_test_svm,predict_y_poly))
'''
yval_svm=np.array(np.asarray(y_train_svm), dtype=np.integer)


plot_decision_regions(X=np.asarray(X_train), 
                      y=yval_svm,
                      clf=clff2, 
                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.title('SVM Decision Region Boundary', size=16)
plt.show()


def popup(message, color):
    import tkinter as tk
    master = tk.Tk()
    msg = tk.Message(master, text = message)
    msg.config(bg=color, anchor='center', justify='center',aspect=480,font=('Arial', 9),relief='raised')

    w = 80 # width for the Tk root
    h = 50 # height for the Tk root

    # get screen width and height
    ws = master.winfo_screenwidth() # width of the screen
    hs = master.winfo_screenheight() # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)

    # set the dimensions of the screen 
    # and where it is placed
    master.geometry('%dx%d+%d+%d' % (w, h, x, y))

    #master.geometry('350x200')
    msg.pack()
    tk.mainloop()

'''
