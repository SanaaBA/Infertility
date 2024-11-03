#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[3]:


df = pd.read_csv('26NOV.csv', sep = ';')


# In[5]:


for i, row in df.iterrows():
    if 0 <= row['Medical expert'] <= 2:
        df.loc[i,'Medical expert'] =0
    if row['Medical expert'] >=3:
        df.loc[i, 'Medical expert'] =1


# In[6]:


df.groupby('Medical expert').size()


# In[7]:


df=df.drop(columns=['Couple','Pregnancy','Nr. of live birth '])


# In[8]:


df.columns


# In[9]:


Y=df['Medical expert']


# In[10]:


X=df.drop(columns=['Medical expert'])


# In[11]:


X.columns


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, Y,shuffle=True,test_size=0.2,random_state=1, stratify=Y)


# # Normalisation

# In[13]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

# FFMC, DMC, DC, ISI, RH
X_train['g1'] = minmax.fit_transform(np.array(X_train['g1']).reshape(-1,1))


# In[14]:


X_test['g1'] = minmax.transform(np.array(X_test['g1']).reshape(-1, 1))


# In[15]:


X_train['g6'] = minmax.fit_transform(np.array(X_train['g6']).reshape(-1,1))


# In[16]:


X_test['g6'] = minmax.transform(np.array(X_test['g6']).reshape(-1, 1))


# In[17]:


clf = MLPClassifier(hidden_layer_sizes=(5),random_state=1,activation= 'logistic',solver='lbfgs', max_iter=1000,alpha=0.0001,early_stopping=True)


# In[18]:


clf.fit(X_train, y_train)
print('train',clf.score(X_train, y_train))


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


plot_confusion_matrix(clf, X_train, y_train)
plt.show()


# In[21]:


y_pred_test=clf.predict(X_test)
y_pred_train=clf.predict(X_train)
print('train_matrix',classification_report(y_train, y_pred_train))
print('test_matrix',classification_report(y_test, y_pred_test))


# In[22]:


print('train',clf.score(X_test, y_test))


# In[23]:


plot_confusion_matrix(clf, X_test, y_test)
plt.show()


# # Balancing classes

# In[24]:


from imblearn.over_sampling import SMOTE 


# In[25]:


sm = SMOTE(k_neighbors = 2,random_state=42)

X_sm, y_sm = sm.fit_resample(X_train, y_train)

print(f'''Shape of X before SMOTE: {X_train.shape}
Shape of X after SMOTE: {X_sm.shape}''')

print('\nBalance of positive and negative classes (%):')
y_sm.value_counts(normalize=True) * 100


# In[26]:


y_sm.groupby(y_sm).size()


# In[27]:


clf = MLPClassifier(hidden_layer_sizes=(5),random_state=1,activation= 'tanh',solver='lbfgs', max_iter=250,alpha=0.001, learning_rate='constant',early_stopping=True)


# In[28]:


mlp = MLPClassifier(random_state=1,max_iter=250, early_stopping=True)


# In[29]:


from sklearn.model_selection import StratifiedKFold


# In[30]:


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)


# # Hyper-parameters tuning

# In[31]:


parameter_space = {
    'hidden_layer_sizes': [(5), (10),(5,5)],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['sgd', 'adam','lbfgs'],
    'alpha': [0.0001, 0.05, 0.001],
    'learning_rate': ['constant', 'adaptive']

}

model = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=skf, scoring='accuracy')
model.fit(X_sm, y_sm)

print('Best parameters found:\n', model.best_params_)


# In[32]:


# Print the best parameters and the corresponding score
print("Best parameters found: ", model.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(model.best_score_))


# In[33]:


clf.fit(X_sm, y_sm)
print('train',clf.score(X_sm, y_sm))


# In[35]:


from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(clf, X_sm, y_sm, cv=skf, scoring='accuracy')


# In[36]:


cv_results


# In[37]:


cv_results.mean()


# In[38]:


plot_confusion_matrix(clf, X_sm, y_sm)
plt.show()


# In[39]:


print('test',clf.score(X_test, y_test))


# In[40]:


y_pred_test=clf.predict(X_test)
y_pred_train=clf.predict(X_sm)
print('train_matrix',classification_report(y_sm, y_pred_train))
print('test_matrix',classification_report(y_test, y_pred_test))


# In[41]:


plot_confusion_matrix(clf, X_test, y_test)
plt.show()


# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


# Define the parameter grid
param_grid_RF = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2,5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=1)

grid_search_RF = GridSearchCV(estimator=rf, param_grid=param_grid_RF, cv=skf, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search to the data
grid_search_RF.fit(X_sm, y_sm)

# Print the best parameters and the corresponding score
print("Best parameters found: ", grid_search_RF.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(grid_search_RF.best_score_))


# In[46]:


classifier = RandomForestClassifier(max_depth= None, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 50, random_state=1)


# In[47]:


classifier.fit(X_sm, y_sm)


# In[48]:


y_pred_train = classifier.predict(X_sm)


# In[49]:


print('test_matrix',classification_report(y_sm, y_pred_train)) 


# In[50]:


y_pred_RF = classifier.predict(X_test)


# In[51]:


RF_accuracy = accuracy_score(y_test, y_pred_RF)


# In[52]:


RF_accuracy


# In[53]:


print('test_matrix',classification_report(y_test, y_pred_RF)) 


# In[54]:


plot_confusion_matrix(classifier, X_test, y_test)
plt.show()


# In[55]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[56]:


SVM=SVC(random_state=1)


# In[57]:


# Définition de la grille de paramètres
param_grid_svm = {
    'C': [0.1, 1, 10, 100],            # Paramètre de régularisation
    'gamma': [1, 0.1, 0.01, 0.001],    # Paramètre du noyau RBF
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],# Type de noyau
    'degree': [1, 2, 3, 4, 5]
}


# In[58]:


# Initialisation de GridSearchCV
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=skf, verbose=2, n_jobs=-1)

# Exécution de la recherche de grille sur l'ensemble d'entraînement
grid_search_svm.fit(X_sm, y_sm)


# In[59]:


print(f"Meilleurs paramètres : {grid_search_svm.best_params_}")
print(f"Meilleur score : {grid_search_svm.best_score_:.4f}")

# Évaluation sur l'ensemble de test
best_model = grid_search_svm.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Score sur l'ensemble de test : {test_score:.4f}")


# In[ ]:





# In[60]:


svm_classifier = SVC(random_state=42)


# In[61]:


svm_classifier = SVC(degree= 3,C= 100, gamma= 1, kernel= 'linear',random_state=1)


# In[62]:


svm_classifier.fit(X_sm, y_sm)


# In[63]:


y_p_svm=svm_classifier.predict(X_test)


# In[64]:


svm_accuracy = accuracy_score(y_test, y_p_svm)  


# In[65]:


svm_accuracy


# In[66]:


print('test_matrix',classification_report(y_test, y_p_svm)) 


# In[67]:


# Initialize the logistic regression model
log_reg = LogisticRegression()

# Define the parameter grid
param_grid_LR = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}


# In[68]:


# Initialize Grid Search with Stratified K-Fold
grid_search_LR = GridSearchCV(estimator=log_reg, param_grid=param_grid_LR, cv=skf, scoring='accuracy')

# Fit Grid Search
grid_search_LR.fit(X_sm, y_sm)


# In[69]:


# Get the best parameters
best_params = grid_search_LR.best_params_
print(f"Best parameters found: {best_params}")

# Train the model with the best parameters
best_log_reg = grid_search_LR.best_estimator_
best_log_reg.fit(X_sm, y_sm)

# Make predictions on the test data
y_pred = best_log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy_test: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)


# In[72]:


LR=LogisticRegression(solver='newton-cg', C=0.01, penalty='none',max_iter=100, random_state=1)


# In[73]:


LR.fit(X_sm, y_sm)


# In[74]:


y_p=LR.predict(X_test)


# In[75]:


LR_accuracy = accuracy_score(y_test, y_p)  


# In[76]:


LR_accuracy


# In[77]:


print('test_matrix',classification_report(y_test, y_p)) 


# # Models comparison

# In[78]:


models = [
    ('Random Forest', RandomForestClassifier(max_depth= None, max_features='auto', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 50, random_state=1)),
    ('Logistic Regression', LogisticRegression(solver='newton-cg', C=0.01, penalty='none',max_iter=100, random_state=1)),
    ('Artificial Neural Network', MLPClassifier(hidden_layer_sizes=(5),random_state=1,activation= 'tanh',solver='lbfgs', max_iter=250,alpha=0.001, learning_rate='constant',early_stopping=True)),
    ('Support Vector Machine', SVC(degree= 1,C= 100, gamma= 1, kernel= 'poly',random_state=1))
]


# In[79]:


results = []
names = []

for name, mdl in models:
    cv_results = cross_val_score(mdl, X_sm, y_sm, cv=skf, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results} {cv_results.mean()} ({cv_results.std()})")


# In[ ]:




