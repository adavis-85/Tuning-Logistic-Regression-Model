#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import copy as cp
import sklearn as sci
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from matplotlib import pyplot

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score,classification_report

from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

data=pd.read_csv('C:data_with_classes.csv')
data = data.dropna()
first_data_forchart=cp.deepcopy(data)

##Function to one hot encode and recombine the dataframe and return both the new dataframe as well as the 
##categorical column names
def encode_fun(big_d):

    
    categorical_columns=big_d.select_dtypes(include=['object', 'category']).columns
   
    for column in categorical_columns:
        inserts = pd.get_dummies(big_d[column], prefix=column)

        big_d = pd.merge(
            left=big_d,
            right=inserts,
            left_index=True,
            right_index=True,
        )
        big_d = big_d.drop(columns=column)
            
        big_d = pd.merge(
            left=big_d,
            right=target,
            left_index=True,
            right_index=True,
        )

    return big_d,categorical_columns

#calling categorical function.
data,elim_col=encode_fun(data)

#over-sampling
over = SMOTE()
data,target=over.fit_resample(data,target)

#under-sampling
under = RandomUnderSampler()
data,target=under.fit_resample(data,target)

#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25,random_state=2)

#setting model with solver and max iterations to handle concvergence
log_r=LogisticRegression(solver='saga',max_iter=50000)

#tuning weight for minority class then weight for majority class will be 1-weight of minority class
#Setting the range for class weights
weights = np.linspace(0.0,0.99,10)
#specifying all hyperparameters with possible values
param= {'C': [0.0001,0.001,0.01,.1,1.0,10,100], 'penalty': ['l1','l2'],"class_weight":[{0:x ,1:1.0 -x} for x in weights]}
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
#Gridsearch for hyperparameter tuning
model= GridSearchCV(estimator= log_r,param_grid=param,scoring='explained_variance',cv=folds,return_train_score=True)
#train model to learn relationships between x and y
model.fit(X_train,y_train)

zero_param=model.best_params_['class_weight'][0]
one_param=model.best_params_['class_weight'][1]
best_penalty=model.best_params_['penalty']
best_c_value=model.best_params_['C']

#fitting new logistic regression model to best parameters which depend on the scoring indicator
log_r2=LogisticRegression(solver='saga',class_weight={0:zero_param,1:one_param},C=best_c_value,penalty=best_penalty,max_iter=50000)
log_r2.fit(X_train,y_train)

# predict probabilities on Test and take probability for class 1([:1])
y_pred_prob_test = lr2.predict_proba(X_test)[:, 1]
#predict labels on test dataset
y_pred_test = lr2.predict(X_test)
# create onfusion matrix
confuse_m = metrics.confusion_matrix(y_test, y_pred_test)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(confuse_m), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#print out table with specific ratios
print(classification_report(y_test, y_pred_test))


# In[ ]:




