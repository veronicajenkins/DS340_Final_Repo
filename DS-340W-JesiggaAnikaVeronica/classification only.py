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
''''x=x[0][:-2]
seg=list(map(lambda x: abs(x), y))'''
y=labels[1:]

x = np.asarray(x)
y=np.asarray(y)
#print(x)
xl=[]
for i in x:
    xl.append(list(map(lambda i : float(i),i)))
    
#x=list(map(lambda x : float(x),x))
#y=y.reshape(1, 1)
x=x.astype(np.float64).tolist()
y=y.astype(np.float64).tolist()

#----------------------------------------------------------------------------

#getting input for prediction

shannonentropy=float(input("Enter shannon entropy:"))
Mprh=float(input("Enter the median peak rise height:"))
insthr=int(input("Enter instantaneous heart rate:"))

input_list=[shannonentropy,Mprh,insthr]


#-----------------------------------------------------------------------------

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
ans=int(clf.predict([input_list]))
print(ans)
if ans ==0:
    print("Atrial Fibrilation")
elif ans==1:
    print("Bradycardia")
elif ans==2:
    print("Tachycardia")
else:
    print("Normal")
#print('Decision Tree Classifier:',accuracy_score(y_test,y_pred))
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
graph.write_png("dtc2features.png")'''
