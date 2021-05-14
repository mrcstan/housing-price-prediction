import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import linear_model
import matplotlib.pyplot as plt
import time

def process_training_data(fname, cols2remove, winsor, hot_encode=True):
    # Read training data and extract response variable (Sale_Price)
    # Set the PIDs as the indices of the data frames/series
    # When performing operation on data frames/series, pandas will match rows
    # with the same index in the different data frames/series
    dat_train = pd.read_csv(fname)
    dat_train = remove_cols(dat_train, cols2remove)
    dat_train_y = dat_train['Sale_Price']
    dat_train.index = dat_train['PID']
    dat_train_y.index = dat_train['PID']
    dat_train.drop(columns=['PID', 'Sale_Price'], inplace=True)

    # Reorder columns alphabetically
    dat_train.reindex(sorted(dat_train.columns), axis=1)

    # Winsorize columns with upper thresholds obtained from given quantile
    if len(winsor):
        quantile = winsor['quantile']
        winsor_threshold = dat_train.quantile(quantile)
        winsor['threshold'] = winsor_threshold
        dat_train = winsorize(dat_train,  winsor['cols'], winsor_threshold)

    # Remove columns with constant values
    #n_cols0 = dat_train.shape[1]
    #dat_train = dat_train.loc[:, (dat_train != dat_train.iloc[0]).any()]
    #n_cols1 = dat_train.shape[1]
    #print('%d constant columns removed' % (n_cols0 - n_cols1))

    # Remove row when corresponding response is nan
    # Replace nan value in predictor with 0
    dat_train = replace_nan(dat_train, 0, dat_train_y)

    #mean_train_y = dat_train_y.mean()
    #std_train_y = dat_train_y.std()
    # dat_train_y = (dat_train_y - mean_train_y)/std_train_y
    # Take the log of the sales price
    dat_train_y = np.log(dat_train_y)

    #cat_col_train = dat_train.select_dtypes(['object']).columns
    # dat_train[cat_columns] = dat_train[cat_columns].apply(lambda x: x.astype('category'))
    # dat_train[cat_columns] = dat_train[cat_columns].apply(lambda x: x.cat.codes)
    #noncat_col_train = np.setdiff1d(dat_train.columns, cat_col_train)
    #dat_train[noncat_col_train] = dat_train[noncat_col_train].sub(mean_train, axis=1)
    #dat_train[noncat_col_train] = dat_train[noncat_col_train].div(std_train, axis=1)

    # Convert categorical variables to binary variables
    # dat_train_dum = pd.get_dummies(dat_train, drop_first=True)
    if hot_encode:
        dat_train = pd.get_dummies(dat_train, prefix_sep='.')

    return dat_train, dat_train_y, winsor


def process_test_data(fname, cols2remove, winsor, fname_y, hot_encode=True):
    # Read test data
    # Set the PIDs as the indices of the data frames/series
    dat_test = pd.read_csv(fname)
    dat_test = remove_cols(dat_test, cols2remove)
    dat_test.index = dat_test['PID']
    dat_test.drop(columns=['PID'], inplace=True)

    # Remove columns with constant values
    #n_cols0 = dat_test.shape[1]
    #dat_test = dat_test.loc[:, (dat_test != dat_test.iloc[0]).any()]
    #n_cols1 = dat_test.shape[1]
    #print('%d constant columns removed'%(n_cols1 - n_cols0))

    # Reorder columns alphabetically
    dat_test.reindex(sorted(dat_test.columns), axis=1)

    # Winsorize columns with given upper thresholds
    if len(winsor):
        dat_test = winsorize(dat_test, winsor['cols'], winsor['threshold'])

    if len(fname_y):
        dat_test_y = pd.read_csv(fname_y)
        dat_test_y.index = dat_test_y['PID']
        assert (dat_test_y.index == dat_test.index).all(); # Check that the PIDs agree
        dat_test_y.drop(columns=['PID'], inplace=True)
        dat_test_y = dat_test_y.squeeze(); # convert single-column data frame into series
        # Remove row when corresponding response is nan
        # Replace nan value in predictor with 0
        dat_test = replace_nan(dat_test, 0, dat_test_y)

        # dat_test_y = (dat_test_y - mean_train_y)/std_train_y
        dat_test_y = np.log(dat_test_y)
    else:
        dat_test_y = []
        dat_test = replace_nan(dat_test, 0, [])

    # scale non-categorical variables of test data with the training mean and std
    #cat_col_test = dat_test.select_dtypes(['object']).columns
    #noncat_col_test = np.setdiff1d(dat_test.columns, cat_col_test)
    #dat_test[noncat_col_test] = dat_test[noncat_col_test].sub(mean_train[noncat_col_test], axis=1)
    #dat_test[noncat_col_test] = dat_test[noncat_col_test].div(std_train[noncat_col_test], axis=1)

    # Convert categorical variables to binary variables
    if hot_encode:
        dat_test = pd.get_dummies(dat_test, prefix_sep='.')

    return dat_test, dat_test_y

def replace_nan(dat, fill_val, dat_y):
    # dat: data frame
    # fill_val: replace nan values in data frame with fill val
    # dat_y: must be series

    if len(dat_y):
        assert(len(dat_y) == len(dat))
        # Remove entire row if the response is nan
        nan_y = dat_y.isna(); # indices with nan response
        if nan_y.any():
            print('Dropping %d rows due to nan response'%sum(nan_y))
            nan_y = dat_y.index[nan_y]
            dat_y.drop(index=nan_y, inplace=True)
            dat.drop(index=nan_y, inplace=True)

    # Rows with NA
    # dat_nan = dat_train[dat.isna().any(axis=1)]
    # print(dat_nan)
    # Columns with NA
    dat_col_nan = dat.columns[dat.isna().any(axis=0)]
    if dat_col_nan.size > 0:
        #print('Columns with NA = ', dat_col_nan)
        dat.fillna(fill_val, inplace=True)
        #dat_col_nan = dat.columns[dat_test.isna().any(axis=0)]
        #print('After replacing nan with 0, columns with NA = ', dat_col_nan)

    return dat

def remove_cols(df, cols2remove):
    cols2remove_valid = np.intersect1d(cols2remove, df.columns)
    #print(cols2remove_valid)
    return df.drop(columns=cols2remove_valid)

def winsorize(df, winsor_cols, winsor_threshold):
    cols = np.intersect1d(winsor_cols, df.columns)
    df[cols] = df[cols].clip(upper=winsor_threshold, axis=1)
    return df

def enforce_data_frame_consistency(dat1, dat2):
    # Add column names of dat1 that are not in dat2 to dat2 and set the values to 0
    missing_predictors = np.setdiff1d(dat1.columns, dat2.columns)
    dat2[missing_predictors] = pd.DataFrame([np.zeros(len(missing_predictors))], index=dat2.index)

    # Add column names of dat2 that are not in dat1 to dat1 and set the values to 0
    missing_predictors = np.setdiff1d(dat2.columns, dat1.columns)
    dat1[missing_predictors] = pd.DataFrame([np.zeros(len(missing_predictors))], index=dat1.index)

    # Reorder columns. This is important since sklearn reads the
    # pandas data frame as a matrix
    #dat1.columns = dat1.sort_index(axis=1).columns; #NOTE: this does not re-order the columns
    #dat2.columns = dat2.sort_index(axis=1).columns
    dat1 = dat1.reindex(sorted(dat1.columns), axis=1)
    dat2 = dat2.reindex(sorted(dat2.columns), axis=1)
    assert (dat1.columns == dat2.columns).all()

    return dat1, dat2

def standardize_data(dat1, mean1, std1):
    if isinstance(dat1, pd.DataFrame):
        # Standardize each column of data frame dat1 with its mean and standard deviation
        # if mean1 or std1 is nan
        if not isinstance(mean1, pd.Series):
            mean1 = dat1.mean();  # ignores nan by default
        if not isinstance(std1, pd.Series):
            std1= dat1.std();  # denominator = sample size-1, ignores nan by default
            std1[np.abs(std1) < 1e-10] = 1.0; # replace std1 by 1 if it is close to 0
        dat1 = dat1.sub(mean1, axis=1)
        dat1 = dat1.div(std1, axis=1)
    elif isinstance(dat1, pd.Series):
        if np.isnan(mean1):
            mean1 = dat1.mean()
        if np.isnan(std1):
            std1 = dat1.std()
            if abs(std1) < 1e-10:
                std1 = 1.0
        dat1 = (dat1 - mean1)/std1
    else:
        raise Exception('Must be either pandas data frame or series')

    return dat1, mean1, std1


def calc_rmse(pred_y, ref_y, std_y):
    abs_diff = np.abs(pred_y-ref_y)
    return np.sqrt(np.mean(abs_diff**2))*std_y


def model_rmse_for_multi_parameters(dat_all, dat_all_y, std_y, test_ids, model, params):
    print('Model = ', model)
    n_sets = test_ids.shape[0]
    n_params = len(params)
    all_rmse = np.ones((n_sets, n_params))*np.nan
    all_elapsed_t = np.ones((n_sets, n_params))*np.nan
    dat_all = dat_all.reset_index()
    for ii in range(n_sets):
        print('Set ', ii+1, ' out of ', n_sets)
        # must make sure that row indices of dat_all are labelled with consecutive integers starting from 0
        dat_train = dat_all.drop(index=test_ids[ii, ::]);
        dat_train_y = dat_all_y.drop(index=test_ids[ii, ::])
        dat_test = dat_all.iloc[test_ids[ii, ::], ]
        dat_test_y = dat_all_y.iloc[test_ids[ii, ::]]

        for jj in range(n_params):
            start_time = time.time()
            if model == 'Linear':
                # must normalize predictors to obtain more accurate result
                # lassoCV = linear_model.LassoCV(normalize=True, alphas=10**(np.linspace(-6, -3, 100)),
                #                               random_state=0, max_iter=10000)
                # reg = lassoCV.fit(dat_train, dat_train_y)
                # print('Best alpha = ', reg.alpha_)
                '''
                lasso = linear_model.Lasso(normalize=True, alpha=params[jj], max_iter=10000)
                lasso.fit(dat_train, dat_train_y)
                pred_y = lasso.predict(dat_test)
                # removed_predictors = dat_train.columns[np.where(reg.coef_ == 0)]
                # np.savetxt('Lasso removed predictors.csv', removed_predictors, fmt='%s', delimiter='\n')
                '''
                # predictors are already normalized
                elasticNet = linear_model.ElasticNet(normalize=False, alpha=params[jj], max_iter=10000, random_state=0)
                elasticNet.fit(dat_train, dat_train_y)
                pred_y = elasticNet.predict(dat_test)
            elif model == 'Xgboost':
                dtrain = xgb.DMatrix(dat_train, label=dat_train_y)
                param = {'max_depth': 6, 'eta': params[jj], 'subsample': 0.5, 'gamma': 0.0,
                         'objective': 'reg:squarederror',
                         'nthread': 4}
                num_round = 5000
                bst = xgb.train(param, dtrain, num_round)
                dtest = xgb.DMatrix(dat_test)
                pred_y = bst.predict(dtest)
            else:
                raise Exception('Unknown model '+ model)

            all_rmse[ii, jj]  = calc_rmse(pred_y, dat_test_y, std_y)
            all_elapsed_t[ii, jj] = time.time() - start_time
            print('Param = ', params[jj], ', RMSE = ', all_rmse[ii, jj], ', elapsed time = ', all_elapsed_t[ii, jj])

    return all_rmse, all_elapsed_t


# User inputs ----------------------------------------------------------------------------------------------------------

param_best = {'Linear': 0.0082864, 'Xgboost': 0.025}
file_train = 'train.csv'
file_test = 'test.csv'
file_test_y = 'test_y.csv'
#file_test_y = []

# Feature selection - Input data columns to be removed
#cols2remove = []
#cols2remove = ['Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature',
#              'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude'];  # suggested by instructors
cols2remove = ['BsmtFin_SF_2', 'Bsmt_Half_Bath', 'Condition_2', 'Land_Slope',
              'Low_Qual_Fin_SF', 'Misc_Feature', 'Mo_Sold', 'Pool_Area', 'Pool_QC',
              'Roof_Matl','Screen_Porch', 'Street', 'Utilities', 'Year_Sold']; # corr <= 0.1
# Winsorization for ElasticNet only
#winsor = []
winsor = {'cols': ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF",
                   "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF",
                   "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"],
          'quantile': 0.92}

# ----------------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns', 10)

''''''
'''
## Feature selections using correlation between predictors and response 
dat_all, dat_all_y, winsor = process_training_data('Ames_data.csv', [], [])
dat_all, mean_all, std_all = standardize_data(dat_all, np.nan, np.nan)
dat_all_y, mean_all_y, std_all_y = standardize_data(dat_all_y, np.nan, np.nan)
Xy_corr = dat_all.corrwith(dat_all_y)
Xy_corr = Xy_corr.abs().sort_values(ascending=True)
corr_thres = 0.1
Xy_corr_small = Xy_corr.loc[Xy_corr <= corr_thres]
Xy_corr_large = Xy_corr.loc[Xy_corr > corr_thres]
Xy_corr_small_vars = Xy_corr_small.index.map(lambda x: x.rsplit('.')[0]); # # remove the characters after . for one-hot-encoded predictors
Xy_corr_large_vars = Xy_corr_large.index.map(lambda x: x.rsplit('.')[0]); # # remove the characters after . for one-hot-encoded predictors
print('Predictors with small correlation with response')
small_corr_vars = np.setdiff1d(Xy_corr_small_vars, Xy_corr_large_vars)
print(repr(small_corr_vars))
'''
'''
## Best parameter selection
cand_params = {'Linear': 10**np.linspace(-2.5, -1, 50), 'Xgboost': np.array([0.025, 0.05])}
model_stats = pd.DataFrame({'Model':[], 'Test_set':[], 'RMSE':[],
                            'Elapsed_t':[], 'Best_param':[]})

dat_all, dat_all_y, winsor = process_training_data('Ames_data.csv', cols2remove, winsor)
dat_all, mean_all, std_all = standardize_data(dat_all, np.nan, np.nan)
dat_all_y, mean_all_y, std_all_y = standardize_data(dat_all_y, np.nan, np.nan)
test_ids = np.transpose(np.genfromtxt('project1_testIDs.dat').astype('int64')-1)
#test_ids = test_ids[:2, :]; # DEBUG!
# must make sure that row indices of dat_all are labelled with consecutive integers starting from 0
dat_all.reset_index(drop=True, inplace=True)
dat_all_y.reset_index(drop=True, inplace=True)
all_rmse = {}

print('X data shape = ', dat_all.shape)
'''
'''
## Get the best ElasticNet model parameter by trying multiple parameters
all_rmse['Linear'], all_elapsed_t = model_rmse_for_multi_parameters(dat_all, dat_all_y, std_all_y, test_ids,
                                                                    'Linear', cand_params['Linear'])
#print(all_rmse['Linear'])
np.savetxt('_Elasticnet all rmse.csv', all_rmse['Linear'])
np.savetxt('_Elasticnet all elapsed t.csv', all_elapsed_t)
#inds = (all_rmse['Linear'][:5, :] <= 0.125).all(axis = 0) & (all_rmse['Linear'][5:, :] <= 0.135).all(axis=0)
# ind_min = np.where(inds)[0]
max_rmse = all_rmse['Linear'].max(axis=0)
ind_min = max_rmse.argmin()
param_best['Linear'] = cand_params['Linear'][ind_min]
print('Best linear model parameter = ', param_best['Linear'])

stats = pd.DataFrame({'Model': 'Linear', 'Test_set': (np.arange(test_ids.shape[0])+1).astype('uint64'),
                      'RMSE': all_rmse['Linear'][:, ind_min],
                      'Elapsed_t': all_elapsed_t[:, ind_min], 'Best_param': param_best['Linear']})
model_stats = model_stats.append(stats)
model_stats.reset_index(drop=True, inplace=True)
print(model_stats)
model_stats.to_csv('_Elasticnet model stats.csv', header=True, sep=',')

fig, ax = plt.subplots()
lines = ax.plot(cand_params['Linear'],
                np.transpose(np.vstack((all_rmse['Linear'], max_rmse))))
ax.set_xlabel('Alpha')
ax.set_ylabel('RMSE')
labels = (np.arange(test_ids.shape[0])+1).astype('str')
labels = np.append(labels, 'max RMSE')
ax.legend(lines, labels, loc='upper left')

'''
'''
## Get the best Xgboost model parameter by trying multiple parameters
all_rmse['Xgboost'], all_elapsed_t = model_rmse_for_multi_parameters(dat_all, dat_all_y, std_all_y, test_ids,
                                                                    'Xgboost', cand_params['Xgboost'])
np.savetxt('_Xgboost all rmse.csv', all_rmse['Xgboost'])
np.savetxt('_Xgboost all elapsed t.csv', all_elapsed_t)
max_rmse = all_rmse['Xgboost'].max(axis=0)
ind_min = max_rmse.argmin()
param_best['Xgboost'] = cand_params['Xgboost'][ind_min]
print('Best Xgboost model parameter = ', param_best['Xgboost'])

stats = pd.DataFrame({'Model': 'Xgboost', 'Test_set': (np.arange(test_ids.shape[0]) + 1).astype('uint64'),
                      'RMSE': all_rmse['Xgboost'][:, ind_min],
                      'Elapsed_t': all_elapsed_t[:, ind_min], 'Best_param': param_best['Xgboost']})
model_stats = model_stats.append(stats)
model_stats.reset_index(drop=True, inplace=True)
print(model_stats)
model_stats.to_csv('_Xgboost model stats.csv', header=True, sep=',')

fig, ax = plt.subplots()
lines = ax.plot(cand_params['Xgboost'],
                np.transpose(np.vstack((all_rmse['Xgboost'], max_rmse))))
ax.set_xlabel('Eta')
ax.set_ylabel('RMSE')
labels = (np.arange(test_ids.shape[0])+1).astype('str')
labels = np.append(labels, 'max RMSE')
ax.legend(lines, labels, loc='upper left')


plt.show()
'''
# ----------------------------------------------------------------------------------------------------------------------
print('===============================================================================================================')
print('Training file = ', file_train)
print('Test file = ', file_test)

# ----------------------------------------------------------------------------------------------------------------------
print('-----------------------------------------------------------------------------------------------------------')
## ElasticNet model
dat_train, dat_train_y, winsor = process_training_data(file_train, cols2remove, winsor)

dat_test, dat_test_y = process_test_data(file_test, cols2remove, winsor, file_test_y)

dat_train, dat_test = enforce_data_frame_consistency(dat_train, dat_test)

dat_train, mean_dat_train, std_dat_train = standardize_data(dat_train, np.nan, np.nan)
dat_train_y, mean_dat_train_y, std_dat_train_y = standardize_data(dat_train_y, np.nan, np.nan)

dat_test, tmp, tmp = standardize_data(dat_test, mean_dat_train, std_dat_train)
if len(dat_test_y):
    dat_test_y, tmp, tmp = standardize_data(dat_test_y, mean_dat_train_y, std_dat_train_y)

print('Training ElasticNet model with alpha = ', param_best['Linear'])
print('Training data shape = ', dat_train.shape)
print('Test data shape = ', dat_test.shape)

start_time = time.time()
# Predictors have been standardized/normalized
elasticNet = linear_model.ElasticNet(normalize=False, alpha=param_best['Linear'], max_iter=10000, random_state=0)
elasticNet.fit(dat_train, dat_train_y)
print('Predict sales price with ElasticNet model')
pred_y = elasticNet.predict(dat_test)
# np.savetxt('ElasticNet removed predictors for Set 2.csv',
#           dat_train.columns[np.where(elasticNet.coef_ == 0)[0]], fmt='%s', delimiter='\n')
if len(dat_test_y) == len(pred_y):
    rmse = calc_rmse(pred_y, dat_test_y, std_dat_train_y)
    print('Elasticnet RMSE = ', rmse)
else:
    print('No test response data provided to test model accuracy')
print('Elasticnet elapsed time = ', time.time() - start_time)
pred_expy = pd.Series(np.exp((pred_y * std_dat_train_y + mean_dat_train_y)),
                      name='Sale_Price', index=dat_test.index)
pred_expy.round(2).to_csv('ElasticNet_predictions.csv', header=True, sep=',')

# ----------------------------------------------------------------------------------------------------------------------
print('-----------------------------------------------------------------------------------------------------------')
## Xgboost model
dat_train, dat_train_y, [] = process_training_data(file_train, cols2remove, [])

dat_test, dat_test_y = process_test_data(file_test, cols2remove, [], file_test_y)

dat_train, dat_test = enforce_data_frame_consistency(dat_train, dat_test)

dat_train, mean_dat_train, std_dat_train = standardize_data(dat_train, np.nan, np.nan)
dat_train_y, mean_dat_train_y, std_dat_train_y = standardize_data(dat_train_y, np.nan, np.nan)

dat_test, tmp, tmp = standardize_data(dat_test, mean_dat_train, std_dat_train)
if len(dat_test_y):
    dat_test_y, tmp, tmp = standardize_data(dat_test_y, mean_dat_train_y, std_dat_train_y)

print('Training Xgboost model with eta = ', param_best['Xgboost'])
print('Training data shape = ', dat_train.shape)
print('Test data shape = ', dat_test.shape)
# Prediction with best Xgboost model
param = {'max_depth': 6, 'eta': param_best['Xgboost'], 'subsample': 0.5, 'gamma': 0.0,
         'objective': 'reg:squarederror',
         'nthread': 4}
num_round = 5000
start_time = time.time()
dtrain = xgb.DMatrix(dat_train, label=dat_train_y)
bst = xgb.train(param, dtrain, num_round)
dtest = xgb.DMatrix(dat_test)
print('Predict sales price with Xgboost model')
pred_y = bst.predict(dtest)
if len(dat_test_y) == len(pred_y):
    rmse = calc_rmse(pred_y, dat_test_y, std_dat_train_y)
    print('Xgboost RMSE = ', rmse)
else:
    print('No test response data provided to test model accuracy')
print('Xgboost elapsed time = ', time.time() - start_time)

pred_expy = pd.Series(np.exp((pred_y * std_dat_train_y + mean_dat_train_y)),
                      name='Sale_Price', index=dat_test.index)
pred_expy.round(2).to_csv('Xgboost_predictions.csv', header=True, sep=',')
