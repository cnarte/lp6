# %%
import numpy as np
import pandas as pd

# %%
data_url = "data/housing.xls"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# %%
raw_df

# %%
# from sklearn.datasets import load_boston
# boston = load_boston()

# %%
data = pd.DataFrame(raw_df)

# %%
data.head()

# %%
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',"PRICE"]

# %%


# %%
data.head()

# %%
data.to_csv("data/boston.csv",index=False)

# %%
print(data.shape)

# %%
data.isnull().sum()

# %% [markdown]
# No null values in the dataset, no missing value treatement needed

# %%
data.describe()

# %% [markdown]
# This is sometimes very useful, for example if you look at the CRIM the max is 88.97 and 75% of the value is below 3.677083 and mean is 3.613524 so it means the max values is actually an outlier or there are outliers present in the column

# %%
data.info()

# %% [markdown]
# <a id = 'visual'></a>
# # Visualisation

# %%
import seaborn as sns
sns.distplot(data.PRICE)

# %% [markdown]
# The distribution seems normal, has not be the data normal we would have perform log transformation or took to square root of the data to make the data normal. Normal distribution is need for the machine learning for better predictiblity of the model

# %%
sns.boxplot(data.PRICE)

# %% [markdown]
# <a id = 'corr'></a>
# ### Checking the correlation of the independent feature with the dependent feature
# 
# Correlation is a statistical technique that can show whether and how strongly pairs of variables are related.An intelligent correlation analysis can lead to a greater understanding of your data

# %%
correlation = data.corr()
correlation.loc['PRICE']

# %%
import matplotlib.pyplot as plt
fig,axes = plt.subplots(figsize=(15,12))
sns.heatmap(correlation,square = True,annot = True)

# %% [markdown]
# By looking at the correlation plot LSAT is negatively correlated with -0.75 and RM is positively correlated to the price and PTRATIO is correlated negatively with -0.51

# %%
plt.figure(figsize = (20,5))
features = ['LSTAT','RM','PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = data[col]
    y = data.PRICE
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')

# %% [markdown]
# <a id = 'split'></a>
# ### Splitting the dependent feature and independent feature 

# %%
X = data.iloc[:,:-1]
y= data.PRICE

# %%
!pip install mord

# %%
!pip install tabulate

# %%
import numpy as np
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from mord import OrdinalRidge
from tabulate import tabulate



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Regression
poly_reg = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
poly_reg.fit(X_train, y_train)
y_pred_poly = poly_reg.predict(X_test)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

# Lasso Regression
lasso_reg = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

# Partial Least Squares Regression
pls_reg = make_pipeline(StandardScaler(), PLSRegression(n_components=5))
pls_reg.fit(X_train, y_train)
y_pred_pls = pls_reg.predict(X_test)
pls_mse = mean_squared_error(y_test, y_pred_pls)
pls_r2 = r2_score(y_test, y_pred_pls)

# Ordinal Regression
ordinal_reg = OrdinalRidge(alpha=0.1)
ordinal_reg.fit(X_train, y_train)
y_pred_ordinal = ordinal_reg.predict(X_test)
ordinal_mse = mean_squared_error(y_test, y_pred_ordinal)
ordinal_r2 = r2_score(y_test, y_pred_ordinal)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)


table = [
    ["Polynomial Regression", poly_mse, poly_r2],
    ["Lasso Regression", lasso_mse, lasso_r2],
    ["Partial Least Squares Regression", pls_mse, pls_r2],
    ["Ordinal Regression", ordinal_mse, ordinal_r2],
    ["Linear Regression", linear_mse, linear_r2]
]


headers = ["Model", "Mean Squared Error", "R2 Score"]
print(tabulate(table, headers, tablefmt="grid"))


# %% [markdown]
# <a id  = 'NN'></a>
# ## Neural Networks

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
import keras
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential

model = Sequential()

model.add(Dense(128,activation  = 'relu',input_dim =13))
model.add(Dense(64,activation  = 'relu'))
model.add(Dense(32,activation  = 'relu'))
model.add(Dense(16,activation  = 'relu'))
# model.add(Dense(8,activation  = 'relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')

# %%
model.summary()


# %%
!pip install graphviz

# %%
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# %%
model.fit(X_train, y_train, epochs = 100)

# %% [markdown]
# <a id = 'eval'></a>
# ### Evaluation of the model

# %%
y_pred_nn = model.predict(X_test)
nn_mse = mean_squared_error(y_test, y_pred_nn)

# %%
from sklearn.metrics import r2_score
nn_r2 = r2_score(y_test, y_pred_nn)
print(nn_r2)

# %%
table = [
    ["Polynomial Regression", poly_mse, poly_r2],
    ["Lasso Regression", lasso_mse, lasso_r2],
    ["Partial Least Squares Regression", pls_mse, pls_r2],
    ["Ordinal Regression", ordinal_mse, ordinal_r2],
    ["Linear Regression", linear_mse, linear_r2],
    ["Neural Network Regression Model",nn_mse,nn_r2 ]
]


headers = ["Model", "Mean Squared Error", "R2 Score"]
print(tabulate(table, headers, tablefmt="markdown"))


