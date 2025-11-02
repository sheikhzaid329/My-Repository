import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 

pima = pd.read_csv("Fish/Fish[1].csv")

X = pima[['Weight','Length1','Height','Width']]
y = pima['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

feature_names= ['Weight','Length1','Height','Width']

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
           special_characters=True,feature_names = feature_names,class_names=['Bream','Roach','Whitefish','parkki','perch','Pike','Smelt'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('speciesv1.png')
Image(graph.create_png())

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = ['Weight','Length1','Height','Width'],class_names=['0','1','2','3','4','5','6'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('speciesV2.png')
Image(graph.create_png())


input("Wait for me...")