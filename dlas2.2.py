# %%
!pip install yellowbrick

# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from yellowbrick.classifier import ConfusionMatrix

# %%
df = pd.read_csv("data/letter-recognition.csv", sep = ",")

# %%
df=df.iloc[:,1:]

# %%
df.head()

# %%
names = ['Class',
         'x-box',
         'y-box',
         'width',
         'high',
         'onpix',
         'x-bar',
         'y-bar',
         'x2bar',
         'y2bar',
         'xybar',
         'x2ybr',
         'xy2br',
         'x-ege',
         'xegvy',
         'y-ege',
         'yegvx']

# %%


# %%
X = df.iloc[:, 1 : 17]
Y = df.select_dtypes(include = [object])


# %%
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.20, random_state = 10)

# %%
scaler = StandardScaler()
scaler.fit(X_train)

# %%
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
mlp = MLPClassifier(hidden_layer_sizes = (250, 300), max_iter = 1000000, activation = 'logistic')

# %%
cm = ConfusionMatrix(mlp, classes="A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z".split(','))

# %%
cm.fit(X_train, Y_train.values.ravel())

# %%
cm.score(X_test, Y_test)

# %%
predictions = cm.predict(X_test)

# %%
print("Accuracy: ", accuracy_score(Y_test, predictions))

# %%
print(confusion_matrix(Y_test, predictions))

# %%
print(classification_report(Y_test, predictions, digits=5))

# %%
cm.poof()

# %%


# %%



