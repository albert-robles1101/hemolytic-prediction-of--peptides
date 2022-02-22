#Codigo y datos adaptados del articulo de Plisson:
#https://pubmed.ncbi.nlm.nih.gov/33024236/

import os
import numpy as np
import pandas as pd
import seaborn as sns
import re
import scipy
import sklearn

from os import path
from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from os import path
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, matthews_corrcoef, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

os.chdir('C:/Users/alber/Documents/resultados_finales')
def normalize1(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Normalize test dataset based on model dataset
def normalize2(df, df2):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df2[feature_name].max()
        min_value = df2[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def hemo_prediction_plisson(db):
    HemoPI1_model = pd.read_csv('HemoPI1_model.csv', index_col=0)
    HemoPI1_validation = pd.read_csv('HemoPI1_validation.csv', index_col=0)
    
    cleaned_HemoPI1_model = HemoPI1_model.dropna(axis='columns')
    cleaned_HemoPI1_model = cleaned_HemoPI1_model.drop(['Sequence',' Sequence', 'y_model_2cl'], axis=1)
    cleaned_HemoPI1_validation = HemoPI1_validation.dropna(axis='columns')
    cleaned_HemoPI1_validation = cleaned_HemoPI1_validation.drop(['Sequence',' Sequence', 'y_validation_2cl'], axis=1)
    
    norm_HemoPI1_model = normalize1(cleaned_HemoPI1_model)
#    norm_HemoPI1_validation = normalize1(cleaned_HemoPI1_validation)
    
    # Create correlation matrix
    corr_matrix = norm_HemoPI1_model.corr()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index of feature columns with correlation greater than 0.75
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    # Drop features 
#    trim_HemoPI1_model = norm_HemoPI1_model.drop(norm_HemoPI1_model[to_drop], axis=1)
    
    X_HemoPI1_model = norm_HemoPI1_model
    y_HemoPI1_model = HemoPI1_model['y_model_2cl']
    
#    X_HemoPI1_validation = norm_HemoPI1_validation 
#   y_HemoPI1_validation = HemoPI1_validation['y_validation_2cl'] 
    
    seed=42
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle= True)
    
    models = []
    models.append(('logreg', LogisticRegression(fit_intercept=True)))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('cart', DecisionTreeClassifier(random_state=seed)))
    models.append(('rfc', RandomForestClassifier(random_state=seed, max_depth=5, n_jobs=10)))
    models.append(('gbc', GradientBoostingClassifier(random_state=seed)))
    models.append(('adc', AdaBoostClassifier()))
    models.append(('lda', LinearDiscriminantAnalysis()))
    models.append(('qda', QuadraticDiscriminantAnalysis()))
    models.append(('nb', GaussianNB()))
    models.append(('svc_lr', SVC(kernel="linear", C=0.025)))
    models.append(('svc_rbf', SVC(probability=True)))
    models.append(('svc_poly', SVC(kernel="poly", probability=True)))
    models.append(('svc_sig', SVC(kernel="sigmoid", probability=True)))
    
    #LDA
    model1_hpi1 = LinearDiscriminantAnalysis(solver='svd', tol=0.0001)
    rfecv_model = RFECV(model1_hpi1, step=1, cv=kfold)
    rfecv = rfecv_model.fit(X_HemoPI1_model, y_HemoPI1_model)
    X_HemoPI1_model_RFE = rfecv.transform(X_HemoPI1_model)

    #GBC_26_descriptors
    model2_hpi1 = GradientBoostingClassifier(n_estimators=240, max_depth=4, min_samples_leaf=10, max_features='sqrt', random_state=seed)
    corr_matrix = norm_HemoPI1_model.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    X_HemoPI1_model_trim = norm_HemoPI1_model.drop(norm_HemoPI1_model[to_drop], axis=1)
    
    
    
    #GBC_56_descriptors
    model3_hpi1 = GradientBoostingClassifier(n_estimators=208, max_depth=4, min_samples_leaf=10, max_features='sqrt', random_state=seed)
    
    #Prediction
    database = pd.read_excel(db+'.xlsx')
    database = database.iloc[:,3:59]
    X_total_hpi1 = normalize2(database, cleaned_HemoPI1_model)
    
    ## HemoPI-1 models

    corr_matrix1 = norm_HemoPI1_model.corr()
    upper = corr_matrix1.where(np.triu(np.ones(corr_matrix1.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    X_HemoPI1_model_trim = norm_HemoPI1_model.drop(norm_HemoPI1_model[to_drop], axis=1)
    X_totalAPD_hpi1_trim = X_total_hpi1[list(X_HemoPI1_model_trim.columns)]
    print(X_totalAPD_hpi1_trim.shape)
    
    rfecv_model = RFECV(model1_hpi1, step=1, cv=kfold)
    rfecv = rfecv_model.fit(X_HemoPI1_model, y_HemoPI1_model)
    X_HemoPI1_model_RFE = rfecv.transform(X_HemoPI1_model)
    X_totalAPD_hpi1_RFE = rfecv.transform(X_total_hpi1)

    #Save test set results into dataframe
    # Model 1.1
    model1_hpi1.fit(X_HemoPI1_model_RFE, y_HemoPI1_model)
    class_df = pd.DataFrame(model1_hpi1.predict(X_totalAPD_hpi1_RFE))
    probs_df = pd.DataFrame(model1_hpi1.predict_proba(X_totalAPD_hpi1_RFE))

    model1_hpi1_apd = class_df.merge(probs_df, how='outer', left_index=True, right_index=True)
    model1_hpi1_apd.index = X_total_hpi1.index
    model1_hpi1_apd.columns = ['model1_class_preds', 'model1_probability_0', 'model1_probability_1']

    # Model 1.2
    model2_hpi1.fit(X_HemoPI1_model_trim, y_HemoPI1_model)
    class_df2 = pd.DataFrame(model2_hpi1.predict(X_totalAPD_hpi1_trim))
    probs_df2 = pd.DataFrame(model2_hpi1.predict_proba(X_totalAPD_hpi1_trim))
    
    model2_hpi1_apd = class_df2.merge(probs_df2, how='outer', left_index=True, right_index=True)
    model2_hpi1_apd.index = X_total_hpi1.index
    model2_hpi1_apd.columns = ['model2_class_preds', 'model2_probability_0', 'model2_probability_1']
    
    # Model 1.3
    model3_hpi1.fit(X_HemoPI1_model, y_HemoPI1_model)
    class_df3 = pd.DataFrame(model3_hpi1.predict(X_total_hpi1))
    probs_df3 = pd.DataFrame(model3_hpi1.predict_proba(X_total_hpi1))
    
    model3_hpi1_apd = class_df3.merge(probs_df3, how='outer', left_index=True, right_index=True)
    model3_hpi1_apd.index = X_total_hpi1.index
    model3_hpi1_apd.columns = ['model3_class_preds', 'model3_probability_0', 'model3_probability_1']

    # Merge model dataframes
    models_hpi1 = [model1_hpi1_apd, model2_hpi1_apd, model3_hpi1_apd]

    results_models_hpi1_apd = pd.concat(models_hpi1, axis=1, join='inner')
    results_models_hpi1_apd.index = model1_hpi1_apd.index
    results_models_hpi1_apd

    results_models_hpi1_apd.to_csv(db+'_plisson.csv')
    

#Predicciones    
hemo_prediction_plisson('HAPPENN_7_35_dataset')  

hemo_prediction_plisson('HemoPi1_test_7_35_dataset')  
hemo_prediction_plisson('HemoPi2_test_7_35_dataset')  
hemo_prediction_plisson('HemoPi3_test_7_35_dataset')  

hemo_prediction_plisson('HemoPi1_train_7_35_dataset')  
hemo_prediction_plisson('HemoPi2_train_7_35_dataset')  
hemo_prediction_plisson('HemoPi3_train_7_35_dataset')  
    
    
    
    
    
    
    
    
    
    