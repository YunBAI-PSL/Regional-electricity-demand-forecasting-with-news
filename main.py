#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:15:18 2023

@author: yunbai
"""

import numpy as np
import pandas as pd
import pickle
import time
import os
import lightgbm as lgb
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from utils import evaluate_deterministic, getTrainTest, expand_data
from dataLoader import get_exgoFeats, datapre_lGBM
from joblib import Parallel, delayed
import matplotlib.patches as mpatches
import shap
shap.initjs()

# objective function of lightGBM model with optuna
def lgbObjective(X,Y,trial):
    # leave the data from last month as validation set
    x_train,x_valid = X[X.index<'2022-12-01'],X[(X.index>='2022-12-01')&(X.index<'2023-01-01')]
    y_train,y_valid = Y[Y.index<'2022-12-01'],Y[(Y.index>='2022-12-01')&(Y.index<'2023-01-01')]

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100)
    }
    
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))

    model.fit(x_train, y_train)    
    Y_pres = model.predict(x_valid)
    Y_pres = pd.DataFrame(Y_pres,index=y_valid.index,columns=y_valid.columns)
    Y_pres_all = expand_data(Y_pres,colName='lGBM_forecasts')
    Y_test_all = expand_data(y_valid,colName='Load')
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    errorDf_all = evaluate_deterministic(Y_test_all[['Load','lGBM_forecasts']])
    
    # optimise rmse
    rmse = errorDf_all.mean()[0]
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return rmse

# training process by optuna for params selction
def training_eval(regionName,taskName,hor_key,X,Y,X_cols,Y_cols):
    # rename columns for lightGBM model
    X_cols_simple,Y_cols_simple = ['X{}'.format(i) for i in range(len(X_cols))],['Y{}'.format(i) for i in range(len(Y_cols))]
    X.columns,Y.columns = X_cols_simple,Y_cols_simple
    X_train,X_test,Y_train,Y_test,scaler_x,scaler_y = getTrainTest(X,Y)

    # start training
    t1 = time.time()
    savePath = 'optunadbs/{}/{}/{}'.format(regionName,taskName,hor_key)
    n_trials = 100
    study = optuna.create_study(direction='minimize', storage='sqlite:///'+savePath+'.db')  
    study.optimize(lambda trial: lgbObjective(X_train, Y_train, trial), callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))])
    
    optunaPath = './Results/{}/{}/{}/optuna/'.format(regionName,taskName,hor_key)
    directory = os.path.dirname(optunaPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    study.trials_dataframe().to_csv('./Results/{}/{}/{}/optuna/optuna_results.csv'.format(regionName,taskName,hor_key), index=False)
    best_params = study.best_params
    t2 = time.time()
    print(t2-t1)

    # fitting the model with trained params and forecast
    lgbm = lgb.LGBMRegressor(objective='regression', metric='mse', verbosity=-1, random_state=42,**best_params)
    multi_model = MultiOutputRegressor(lgbm)
    multi_model.fit(X_train,Y_train)
    Y_pres = multi_model.predict(X_test)    
    Y_pres = scaler_y.inverse_transform(Y_pres)
    Y_pres = pd.DataFrame(Y_pres,index=Y_test.index,columns=Y_test.columns)
    Y_pres_all = expand_data(Y_pres,colName='lGBM_forecasts')
    Y_test_all = expand_data(Y_test,colName='Load')
    Y_test_all = Y_test_all.merge(Y_pres_all,left_on='Date', right_on='Date', how='inner')
    errorDf_all = evaluate_deterministic(Y_test_all[['Load','lGBM_forecasts']])

    return best_params,errorDf_all

# forecast with lightGBM
def forecast_lGBM_fullData(regionName,taskName):
    """
    taskName: task names: noText_noEco, noText_withEco, withText_noEco, withText_withEco
    regionName: region names: EastMidlands, WestMidlands, SouthWales, SouthWest, loadNIE, loadIreland
    Ecodata: economic features of GDP, unemployment, and inflation
    textFeat: full News text features
    """
    horizon_dict = dict()  #save detailed forecasting errors on each horizon
    ML_error_df = pd.DataFrame()  # make the summarising table
    Horizons = []
    lGBM_RMSE,lGBM_MAPE = [],[]
    best_params_dict = dict()  # save the selected lgb params of model

    # loading centroids from national news
    textFeat, textCentroids, Ecodata = get_exgoFeats(regionName)

    for horizon in range(1, 31):
        hor_key = 'Horizon_{}'.format(horizon)
        print(hor_key)
        X,Y,XY_table_h = datapre_lGBM(regionName,horizon,Ecodata,textFeat)
        X_cols,Y_cols = list(X.columns),list(Y.columns)
        if 'noEco' in taskName:
            # del the last three economic indexes
            X = X.iloc[:,:-3]
            X_cols = X_cols[:-3]

        if taskName == 'noText_noEco' or taskName == 'noText_withEco':
            best_params,errorDf_all = training_eval(regionName,taskName,hor_key,X,Y,X_cols,Y_cols):
    
        # train with optuna with the tasks of: withText_noEco or withText_withEco
        if taskName == 'withText_noEco' or taskName == 'withText_withEco':
            # prepare data for text forecast model
            XY_table_h = XY_table_h[X_cols+Y_cols]
            # merge the centroids text data
            XY_table_h = XY_table_h.merge(textCentroids,left_on='Date',right_on='Date',how='inner')     
            X_cols = X_cols + list(textCentroids.columns)
            XY_table_h = XY_table_h.dropna()
            X_text,Y_text = XY_table_h[X_cols],XY_table_h[Y_cols]

            best_params,errorDf_all = training_eval(regionName,taskName,hor_key,X_text,Y_text,X_cols,Y_cols):
 
        ########### save results
        best_params_dict[hor_key] = best_params
                
        # save errorDf_all
        errorDf_allPath = './Results/{}/{}/{}/errorDf_all.pkl'.format(regionName,taskName,hor_key)
        directory = os.path.dirname(errorDf_allPath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(errorDf_allPath,'wb') as f1:
            pickle.dump(errorDf_all,f1)
        
        # save the results
        horizon_dict[hor_key] = errorDf_all
        Horizons.append(hor_key)
        lGBM_RMSE.append(errorDf_all.mean()[0])
        lGBM_MAPE.append(errorDf_all.mean()[1])
        
    ML_error_df['Horizons'] = Horizons
    ML_error_df['lGBM_RMSE'],ML_error_df['lGBM_MAPE'] = lGBM_RMSE, lGBM_MAPE
    
    # save the best params for model
    paramPath = './Results/{}/{}/lgbm_params.pkl'.format(regionName,taskName)
    directory = os.path.dirname(paramPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(paramPath,'wb') as f2:
        pickle.dump(best_params_dict,f2) 
    
    # save the averaged error table
    ML_errorsPath = './Results/{}/{}/errorTable.csv'.format(regionName,taskName)   
    ML_error_df.to_csv(ML_errorsPath,index=False)

# training for shap values
def training_shap(regionName):
    # using the results of task: withText_withEco
    paramPath = './Results/{}/withText_withEco/lgbm_params.pkl'.format(regionName)
    with open(paramPath,'rb') as f:
        best_params_dict = pickle.load(f) 

    # loading centroids from national news
    textFeat, textCentroids, Ecodata = get_exgoFeats(regionName)

    # define shap function
    def compute_shap_values(model, X):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        return shap_values

    shap_dict = dict()
    for horizon in range(1, 31):
        hor_key = 'Horizon_{}'.format(horizon)
        print(hor_key)
        X,Y,XY_table_h = datapre_lGBM(regionName,horizon,Ecodata,textFeat)
        X_cols,Y_cols = list(X.columns),list(Y.columns)
        
        # prepare data for text forecast model
        XY_table_h = XY_table_h[X_cols+Y_cols]
        # merge the centroids text data
        XY_table_h = XY_table_h.merge(textCentroids,left_on='Date',right_on='Date',how='inner')     
        X_cols = X_cols + list(textCentroids.columns)
        XY_table_h = XY_table_h.dropna()
        X_text,Y_text = XY_table_h[X_cols],XY_table_h[Y_cols]
        X_cols_simple,Y_cols_simple = ['X'+str(i) for i in range(len(X_cols))],['Y'+str(i) for i in range(len(Y_cols))]
        X_text.columns,Y_text.columns = X_cols_simple,Y_cols_simple
        X_train_text,X_test_text,Y_train_text,Y_test_text,scaler_x_text,scaler_y_text = getTrainTest(X_text,Y_text)
        
        # retrain the model with best params
        best_params_text = best_params_dict[hor_key]
        lgbm_text = lgb.LGBMRegressor(objective='regression', metric='mse', verbosity=-1, random_state=42,**best_params_text)
        multi_model_text = MultiOutputRegressor(lgbm_text)
        multi_model_text.fit(X_train_text,Y_train_text)

        # get the SHAP values
        shaps = Parallel(n_jobs=-1)(delayed(compute_shap_values)(estimator, X_train_text) for estimator in multi_model_text.estimators_)
        shap_dict[hor_key] = [shaps,X_train_text]

    # save the best params for model without text
    shapPath = './Results/{}/ShapValues.pkl'.format(regionName)
    directory = os.path.dirname(shapPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(shapPath,'wb') as f:
        pickle.dump(shap_dict,f) 

# analysing and plotting shap values
def Shap_analysis(regionName):
    shapPath = './Results/{}/ShapValues.pkl'.format(regionName)
    with open(shapPath,'rb') as f:
        shap_dict = pickle.load(f) 
    
    features = ['GDP','Unemployment','Inflation', 'Military conflicts','Transportation',
                'Travel & leisure', 'Sports events', 'Pandemic control', 'Regional economics',
                'Strikes', 'Family life', 'Election', 'Energy markets']
    
    feat_imp = []
    X_train_all, shaps_all_H = [],[]
    for i in range(1, 31):
        hor_key = 'Horizon_'.format(i)
        shaps = shap_dict[hor_key][0]
        X_train_text = shap_dict[hor_key][1]
        shaps_all_hour = sum(shaps)
        shaps_eco_text = shaps_all_hour[:, -13:]

        # summerise feature importance on all the horizons
        X_train_all.append(X_train_text)
        shaps_all_H.append(shaps_eco_text)

        # get feature importance for each horizon
        mean_abs_shap_values = np.abs(shaps_eco_text).mean(axis=0)
        feat_imp.append(list(mean_abs_shap_values))
    
    feat_imp = pd.DataFrame(feat_imp,columns=features)
    X_train = pd.concat(X_train_all,axis=0)
    shaps_H = np.concatenate(shaps_all_H, axis=0)

    return feat_imp, X_train, shaps_H

def plot_shap(regionName):
    features = ['GDP','Unemployment','Inflation', 'Military conflicts','Transportation',
                'Travel & leisure', 'Sports events', 'Pandemic control', 'Regional economics',
                'Strikes', 'Family life', 'Election', 'Energy markets']
    feat_imp, X_train, shaps_H = Shap_analysis(Name)
    feats_sort = feat_imp.mean(axis=0).nlargest(3).index
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    def plot_beeswarm():
        # plot the beeswarm
        shap.summary_plot(shaps_H, X_train.iloc[:,-13:], # select the last 13 features: 10 for text, 3 for economic
                          feature_names=features,show=False)
        plt.gcf().set_size_inches(3,3) 
        plt.savefig('./Results/Figures/ShapBeeswarm-{}.png'.format{regionName}, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close('all')

    def plot_dependency_top1():
        # plot dependency top1
        shap.dependence_plot(feats_sort[0], shaps_H[:896], # select the rows without nans
                             X_train.iloc[:896,-13:], interaction_index="auto", 
                             feature_names=features,show=False)
        plt.gcf().set_size_inches(3,3)  
        plt.savefig('./Results/Figures/ShapDepenTop1-{}.png'.format{regionName}, dpi=300, bbox_inches='tight', pad_inches=0)

        plt.clf()
        plt.close('all')
    
    def plot_dependency_top2():
        # plot dependency top2
        shap.dependence_plot(feats_sort[1], shaps_H[:896], 
                             X_train.iloc[:896,-13:], interaction_index="auto", 
                             feature_names=features,show=False)
        plt.gcf().set_size_inches(3,3)   
        plt.savefig('./Results/Figures/ShapDepenTop2-{}.png'.format{regionName}, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close('all')
    
    # get the top-3 features
    def plot_feat_horizon():
        plt.rcParams.update({'font.size': 10})
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        fig,ax = plt.subplots(figsize=(3,3))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        horizons = [i for i in range(1,31)]
        f1,f2,f3 = feats_sort[0],feats_sort[1],feats_sort[2]
        plt.plot(horizons, feat_imp[f1], color="#ff0059", label=f1,linewidth=1.5)  
        plt.plot(horizons, feat_imp[f2], color="#9e29ad", label=f2,linewidth=1.5) 
        plt.plot(horizons, feat_imp[f3], color="#0170e8", label=f3,linewidth=1.5) 
        plt.xlabel("Horizons (Day)")
        plt.ylabel("SHAP values")
        
        red_patch = mpatches.Patch(color='#ff0059', label=f1[0])
        purple_patch = mpatches.Patch(color='#9e29ad', label=f2[0])
        blue_patch = mpatches.Patch(color='#0170e8', label=f3[0])

        fig.legend(handles=[red_patch, purple_patch, blue_patch], 
                   loc='lower center', 
                   ncol=3, 
                   bbox_to_anchor=(0.5, -0.15),
                   handlelength=1)
        plt.savefig('./Results/Figures/ShapHorizon-{}.png'.format{regionName},
                    dpi=300,bbox_inches='tight', pad_inches=0.0)
        plt.clf()
        plt.close('all')

    plot_beeswarm()
    plot_dependency_top1()
    plot_dependency_top2()
    plot_feat_horizon()

if __name__ == '__main__':
    regions = ['EastMidlands','WestMidlands','SouthWest','SouthWales','loadNIE','loadIreland']
    tasks = ['noText_noEco', 'noText_withEco', 'withText_noEco', 'withText_withEco']
    
    # choose your interested region and task, for example in East Midlands, training and analysing with both text and economic factors:
    region,task = 'EastMidlands','withText_withEco'

    # training and evaluating
    forecast_lGBM_fullData(region,task)

    # if you want to analyse the shap values
    training_shap(region)
    plot_shap(region)




