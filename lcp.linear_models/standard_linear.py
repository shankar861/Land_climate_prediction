from scipy.io import loadmat
import pickle
from sklearn.linear_model import Lasso,LassoCV,LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from math import sqrt
import math
import numpy as np
import copy
import os
from sklearn import model_selection
import properties

def lasso_regression(X_Train,X_Test,Y_Train,Y_Test):

    X_Tr=copy.deepcopy(X_Train)
    X_Te=copy.deepcopy(X_Test)
    Y_Tr=copy.deepcopy(Y_Train)
    Y_Te=copy.deepcopy(Y_Test)
    #lasso_cv = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto',
                       #max_iter=10000, tol=0.00001, copy_X=True, cv=10, verbose=False, n_jobs=1, positive=False,
                       #random_state=None, selection='cyclic')
    lasso_cv = LassoCV(alphas=np.linspace(0.01,0.5,25), fit_intercept=True, normalize=False, precompute='auto',
                       max_iter=10000, tol=0.00001, copy_X=True, cv=10, verbose=False, n_jobs=1, positive=False,
                       random_state=None, selection='cyclic')
    lasso_cv.fit(X_Tr, Y_Tr)
    Tr_pred=lasso_cv.predict(X_Tr)
    Te_pred=lasso_cv.predict(X_Te)
    l=lasso_cv.alpha_
    beta=lasso_cv.coef_
    r2_train = r2_score(Y_Tr, Tr_pred)
    r2_test = r2_score(Y_Te, Te_pred)
    rmse_train = sqrt(mean_squared_error(Y_Tr, Tr_pred))
    rmse_test = sqrt(mean_squared_error(Y_Te, Te_pred))
    return (Tr_pred,Te_pred)


def cal_dis(latitude1, longitude1,latitude2, longitude2):
    latitude1 = (math.pi/180.0)*latitude1
    latitude2 = (math.pi/180.0)*latitude2
    longitude1 = (math.pi/180.0)*longitude1
    longitude2= (math.pi/180.0)*longitude2
    R = 20
    temp=math.sin(latitude1)*math.sin(latitude2)+\
         math.cos(latitude1)*math.cos(latitude2)*math.cos(longitude2-longitude1)
    # if repr(temp)>1.0:
    #      temp = 1.0
    d = math.acos(temp)*R
    return d

def alasso(X_Train,X_Test,Y_Train,Y_Test,target):
    X_Tr=copy.deepcopy(X_Train)
    X_Te=copy.deepcopy(X_Test)
    Y_Tr=copy.deepcopy(Y_Train)
    Y_Te=copy.deepcopy(Y_Test)
    [Tr_sample,num_feature]=X_Tr.shape
    dict={'Brazil':np.array([-10.0,310.0]),'Peru':np.array([-5.75,283.0]),'Asia':np.array([-10.0,137.0])}
    target_location=dict[target]
    resolution={332:[10,'lat_lon_10x10.mat'],1257:[5,'lat_lon_5x5.mat'],5881:[2.5,'lat_lon.mat']}
    step=resolution[num_feature][0]
    data=loadmat(resolution[num_feature][1])
    position=data['lat_lon_data']
    [row,column]=position.shape
    lat_block=np.zeros(row)
    lon_block=np.zeros(column)
    for i in range(row):
        lat_block[i]=-90+step*i
    for i in range(column):
        lon_block[i]=0+i*step

    distance=np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            distance[i][j]=cal_dis(lat_block[i],lon_block[j],target_location[0],target_location[1])
    max_value=np.amax(distance)

    for i in range(row):
        for j in range(column):
            distance[i][j]=distance[i][j]/max_value

    weight=np.zeros(num_feature)
    count=-1
    for i in range(row):
        for j in range(column):
            if position[i][j]==0:
                count=count+1
                weight[count]=distance[i][j]

    X_w=np.zeros((Tr_sample,num_feature))
    for i in range(num_feature):
        for j in range(Tr_sample):
            X_w[j][i]=X_Tr[j][i]/weight[i]

    lasso_cv = LassoCV(alphas=np.linspace(0.01,0.5,25), fit_intercept=True, normalize=False, precompute='auto',
                       max_iter=10000, tol=0.00001, copy_X=True, cv=10, verbose=False, n_jobs=1, positive=False,
                       random_state=None, selection='cyclic')

    lasso_cv.fit(X_w,Y_Tr)
    l=lasso_cv.alpha_
    beta=lasso_cv.coef_
    beta_update=np.zeros(beta.shape)
    for i in range(num_feature):
        beta_update[i]=beta[i]/weight[i]

    Tr_pred=X_Tr.dot(beta_update)
    Te_pred=X_Te.dot(beta_update)

    beta=lasso_cv.coef_
    r2_train = r2_score(Y_Tr, Tr_pred)
    r2_test = r2_score(Y_Te, Te_pred)
    rmse_train = sqrt(mean_squared_error(Y_Tr, Tr_pred))
    rmse_test = sqrt(mean_squared_error(Y_Te, Te_pred))
    return (Tr_pred,Te_pred)

def pcr(X_Train,X_Test,Y_Train,Y_Test):
    X_Tr=copy.deepcopy(X_Train)
    X_Te=copy.deepcopy(X_Test)
    Y_Tr=copy.deepcopy(Y_Train)
    Y_Te=copy.deepcopy(Y_Test)
    [Tr_sample,num_feature]=X_Tr.shape
    pca=PCA()
    X_reduced_tr=pca.fit_transform(scale(X_Tr))
    var=pca.explained_variance_ratio_
    a_var=0
    kf_10 = model_selection.KFold(n_splits=10, shuffle=False)
    n = len(X_reduced_tr)
    for i in range(num_feature):
        a_var=a_var+var[i]
        if a_var>0.9:
            num_pc=i
            break
    if num_pc<20:
        num_pc=20
    regr = LinearRegression()
    mse = []
    score = -1 * model_selection.cross_val_score(regr, np.ones((n, 1)), Y_Tr.ravel(), cv=kf_10,
                                                 scoring='neg_mean_squared_error').mean()
    mse.append(score)
    for i in np.arange(1,num_pc):
        kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
        score = -1 * model_selection.cross_val_score(regr, X_reduced_tr[:, :i], Y_Tr.ravel(), cv=kf_10,
                                                     scoring='neg_mean_squared_error').mean()
        mse.append(score)
    val, idx = min((val, idx) for (idx, val) in enumerate(mse))
    X_reduced_te = pca.transform(scale(X_Te))[:, :idx]
    regr = LinearRegression()
    regr.fit(X_reduced_tr[:, :idx], Y_Tr)
    Tr_pred=regr.predict(X_reduced_tr[:, :idx])
    Te_pred = regr.predict(X_reduced_te)
    return (Tr_pred,Te_pred)

os.chdir(properties.Lasso_Data_path)

X=pickle.load(open("X_sep_2x2.p","rb"))
Y=pickle.load(open("Y_sep.p","rb"))

num_climate_model=len(X)
num_target=Y[0][0][0].shape[1]
target_name=['Brazil','Peru','Asia']

rmse=[]
r2=[]
for climate_model in range(num_climate_model):
    num_window=len(X[climate_model])
    temp_rmse=np.zeros((num_window,6))
    temp_r2 = np.zeros((num_window, 6))
    for target in range(num_target):
        for window in range(num_window):
            X_train=X[climate_model][window][0]
            X_test=X[climate_model][window][1]
            Y_train=Y[climate_model][window][0][:,target]
            Y_test=Y[climate_model][window][1][:,target]
            name=target_name[target]
            (Tr_pred_lasso,Te_pred_lasso)=lasso_regression(X_Train=X_train,X_Test=X_test,Y_Train=Y_train,Y_Test=Y_test)
            temp_r2[window][2] = r2_score(Y_train, Tr_pred_lasso)
            # print('r2_train')
            # print(r2_train)
            temp_r2[window][3] = r2_score(Y_test, Te_pred_lasso)
            # print('r2_test')
            # print(r2_test)
            temp_rmse[window][2]= sqrt(mean_squared_error(Y_train, Tr_pred_lasso))
            # print('rmse_train')
            # print(rmse_train)
            temp_rmse[window][3] = sqrt(mean_squared_error(Y_test, Te_pred_lasso))
            # print('rmse_test')
            # print(rmse_test)

            (Tr_pred_alasso,Te_pred_alasso)=alasso(X_Train=X_train,X_Test=X_test,Y_Train=Y_train,Y_Test=Y_test,target=name)

           # print('results of alasso')
            temp_r2[window][4] = r2_score(Y_train, Tr_pred_alasso)
            # print('r2_train')
            # print(r2_train)
            temp_r2[window][5] = r2_score(Y_test, Te_pred_alasso)
            # print('r2_test')
            # print(r2_test)
            temp_rmse[window][4]= sqrt(mean_squared_error(Y_train, Tr_pred_alasso))
            # print('rmse_train')
            # print(rmse_train)
            temp_rmse[window][5] = sqrt(mean_squared_error(Y_test, Te_pred_alasso))
            # print('rmse_test')
            # print(rmse_test)

            (Tr_pred_pcr,Te_pred_pcr)=pcr(X_Train=X_train,X_Test=X_test,Y_Train=Y_train,Y_Test=Y_test)
            print('results of pcr')
            temp_r2[window][0] = r2_score(Y_train, Tr_pred_pcr)
            # print('r2_train')
            # print(r2_train)
            temp_r2[window][1] = r2_score(Y_test, Te_pred_pcr)
            # print('r2_test')
            # print(r2_test)
            temp_rmse[window][0] = sqrt(mean_squared_error(Y_train, Tr_pred_pcr))
            # print('rmse_train')
            # print(rmse_train)
            temp_rmse[window][1] = sqrt(mean_squared_error(Y_test, Te_pred_pcr))
            # print('rmse_test')
            # print(rmse_test)
            # print('lalala')
    rmse.append(temp_rmse)
    r2.append(temp_r2)
pickle.dump( rmse, open( "rmse_brazil_2x2.p", "wb" ) )
pickle.dump( r2, open( "r2_brazil_2x2.p", "wb" ) )



