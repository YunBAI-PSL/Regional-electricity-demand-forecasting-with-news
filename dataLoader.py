import pandas as pd

def get_exgoFeats(Name):
    # load the full text features
    textFeat = pd.read_csv('./datasets/text_features.csv')
    textFeat.Date = pd.to_datetime(textFeat.Date,utc=True)
    textFeat = textFeat.set_index('Date')

    # load the text clustering centroids
    with open('./datasets/hier_clusters_feat.pkl','rb') as f:
        clusters = pickle.load(f)
    textCentroids = {}
    for i in range(1,11):
        kmfeats = clusters[i]
        textfeat_H = textFeat[kmfeats]       
        scaler = StandardScaler()
        scaler.fit(textfeat_H)
        textfeat_H_normalized = scaler.transform(textfeat_H)
        textfeat_H.loc[:, :] = textfeat_H_normalized
        textfeat_H = textfeat_H.dropna()
        textfeat_H = textfeat_H.mean(axis=1).to_frame()  
        textCentroids['C'+str(i)] = list(textfeat_H[0])
    textCentroids = pd.DataFrame(textCentroids,index = textFeat.index)
    textCentroids.index = pd.to_datetime(textCentroids.index,utc=True)

    # load economy features
    if Name != 'loadIreland':
        # Economy data for the UK regions except for Ireland
        Ecodata = pd.read_csv('./datasets/Ecodata.csv')
    else:
        Ecodata = pd.read_csv('./datasets/Ecodata-Ireland.csv')
        
    Ecodata['Date'] = pd.to_datetime(Ecodata['Date'],utc=True)
    Ecodata = Ecodata.set_index('Date') 
    return textFeat, textCentroids, Ecodata

def datapre_lGBM(Name,horizon,Ecodata,textFeat):
    """
    Name: The name of dataset, if Name='4region' then for sum of EastMidlands, WestMidlans, SouthWales, and SouthWest
    Horizon: from 1d to 30d,
    Ecodata,textFeat: Economic and text information
    """
    if Name != '4regions':
        Y_cols = ['target_Hour_'+str(i) for i in range(24)]
        path_XYtable = './datasets/loadData/{}/XYtable_H_{}.pkl'.format(Name, horizon)
        with open(path_XYtable,'rb') as f2:
            XY_table_h = pickle.load(f2)
        XY_table_h = pd.concat([XY_table_h[0],XY_table_h[1]],axis=1)
        X_cols = [col for col in XY_table_h.columns if (col not in Y_cols)]
    else:
        # sum up the Y data, X data for first 24 cols(hour 0 to hour 23)
        # average rest X data
        Y_cols = ['target_Hour_'+str(i) for i in range(24)]
        X_hour_cols = ['Hour_'+str(i) for i in range(24)]
        path_XYtable = './datasets/loadData/EastMidlands/XYtable_H_{}.pkl'.format(horizon)
        with open(path_XYtable,'rb') as f2:
            XY_table_h = pickle.load(f2)
        XY_table_h = pd.concat([XY_table_h[0],XY_table_h[1]],axis=1)

        for s in ['WestMidlands','SouthWales','SouthWest']:
            path_XYtable = './datasets/loadData/{}/XYtable_H_{}.pkl'.format(s, horizon)
            with open(path_XYtable,'rb') as f2:
                N_XY_table_h = pickle.load(f2)
            N_XY_table_h = pd.concat([N_XY_table_h[0],N_XY_table_h[1]],axis=1)
            XY_table_h = XY_table_h + N_XY_table_h       
        
        XY_table_h[X_hour_cols] = XY_table_h[X_hour_cols]/4
        # del the columns with all nan, because the regions don't share same holidays
        XY_table_h = XY_table_h.drop(columns=list(XY_table_h.columns[XY_table_h.isna().all()]))
        X_cols = [col for col in XY_table_h.columns if (col not in Y_cols)]
        
    # add Economic data
    XY_table_h = XY_table_h.merge(Ecodata,left_on='Date',right_on='Date',how='inner')
    X_cols = X_cols + list(Ecodata.columns)
    # add text data before shift the issued and target date
    XY_table_h = XY_table_h.merge(textFeat,left_on='Date',right_on='Date',how='inner')
    
    # split for X and Y data
    XY_table_h['issued_date'] = XY_table_h.index
    XY_table_h['target_date'] = XY_table_h['issued_date'].shift(-horizon)
    XY_table_h = XY_table_h.dropna()
    XY_table_h = XY_table_h.set_index('target_date')
    XY_table_h = XY_table_h.rename_axis('Date')
    X,Y = XY_table_h[X_cols],XY_table_h[Y_cols] 
    return X,Y,XY_table_h



  
    
    
    
    
    
    
    
    
    
    
    
    



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