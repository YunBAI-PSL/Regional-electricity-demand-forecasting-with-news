import pandas as pd
import numpy as np

def evaluate_deterministic(Y):
    stringPred = Y.columns[1]
    stringObs = 'Load'
    dateList = Y.index.tolist()
    Y['horizon'] = [dt.date() for dt in dateList]
    
    Y['WEerror'] = Y[stringPred] - Y[stringObs]
    
    Y['WEerrorSquare'] = Y['WEerror']**2
    Y['WEabsError'] = Y['WEerror'].abs()
    Y['WEmapeError'] = Y['WEabsError']/np.abs(Y[stringObs])*100
 
    dfEval = pd.DataFrame(columns=['RMSE', 'MAPE'])
    dfEval['RMSE'] = Y.groupby('horizon').mean()['WEerrorSquare']**0.5
    dfEval['MAPE'] = Y.groupby('horizon').mean()['WEmapeError']

    return dfEval

def getTrainTest(X,Y):
    X_train, X_test = X[X.index < '2023-01-01'],X[X.index >= '2023-01-01']
    Y_train, Y_test = Y[Y.index < '2023-01-01'],Y[Y.index >= '2023-01-01']
    
    Xcolumns,Ycolumns = X.columns,Y.columns
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y = StandardScaler()
    scaler_y.fit(Y_train)
    
    X_train = scale_data(Xcolumns,scaler_x,X_train)
    X_test = scale_data(Xcolumns,scaler_x,X_test)
    
    Y_train = scale_data(Ycolumns,scaler_y,Y_train)
    # Y_test = scale_data(Ycolumns,scaler_y,Y_test)
   
    return X_train,X_test,Y_train,Y_test,scaler_x,scaler_y

def expand_data(data,colName):
    fore_df = pd.DataFrame()
    for i in range(len(data)):
        temp = data.iloc[i,:]
        temp_index = [data.index[i] + datetime.timedelta(hours=j) for j in range(24)]
        temp = temp.to_frame()
        if type(colName) == list:
            temp.columns = colName
        else:
            temp.columns = [colName]
        temp = temp.reset_index(drop=True)
        temp['Date'] = temp_index
        temp = temp.set_index('Date')
        fore_df = pd.concat([fore_df,temp])
    return fore_df

  
    
    
    
    
    
    
    
    
    
    
    
    

