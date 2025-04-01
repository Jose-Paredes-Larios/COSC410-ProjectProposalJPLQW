import pandas as pd
import numpy as np

def standardize(filepath):
    '''Standardizes the scales across features whose scales go beyond 0-1. Here, Z-score standardization is used:
        Xst=(Xraw - mean)/std
    '''
    data = pd.read_csv(filepath)
    for col in data:
        if col=="age" or col=="trestbps" or col=="thalach" or col=="oldpeak" or col=="chol":
            mean=data[col].mean()
            std=data[col].std()
            data[col] = (data[col] - mean)/std
    return data

def oneHotEncoding(data):
    '''Takes a feature with n categorical lables and formats it into n features with a 0 or 1 label in accordance with the original label.
    '''
    
    for col in data.columns:
        if data[col].nunique() > 2 and col!="age" and col!="trestbps" and col!="thalach" and col!="oldpeak" and col!="chol":
            data=pd.get_dummies(data, columns=[col])

    #the previous operation uses true and false labels. Thus, this is here to turn any instance of true and false into 1s and 0s, respectively, without altering the floating points we found earlier with the standardize function.
    data[data.select_dtypes(include=["bool"]).columns] = data.select_dtypes(include=["bool"]).astype(int)
    return data  

def completeProcessing(filepath, output):
    '''Processeses a data set so that it can be used to train and evaluate a KNN regression model'''
    data=standardize(filepath)
    data=oneHotEncoding(data)
    data.to_csv(output, index=False)  
    
if __name__ == '__main__':
    completeProcessing("UCIDataset.csv","processedUCIDataset.csv")
    print("done")
