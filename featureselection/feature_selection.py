import  pandas as pd
import numpy as np
import shap

def missing_values_table(df):
    """Function to calculate missing values by column"""
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(3)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns
# 1. functions used to eliminate features
def get_low_std_cols(df,threshold=0.0001,verbose=True):
    """function to get the columns of a dataframe with std lt threshold"""
    columns = list(set(df.columns)-set(["id","user_id","label","target"]))
    eli_cols = []
    for col in columns:
        if df[col].std()<threshold:
            eli_cols.append(col)
    if verbose:
        print("{} out of {} with std lt threshold {}".format(len(eli_cols),len(columns),threshold))
    return eli_cols
def get_missing_ratio_cols(df,threshold=0.5,verbose=True):
    """function to get the columns of a dataframe with missing value ratio gt threshold"""
    columns = list(set(df.columns)-set(["id","user_id","label","target"]))
    eli_cols = []
    df_total_lenth = df.shape[0]
    for col in columns:
        if df[col].isnull().sum()/df_total_lenth>threshold:
            eli_cols.append(col)
    if verbose:
        print("{} out of {} with missing value ratio gt threshold {}".format(len(eli_cols),len(columns),threshold))
        print(missing_values_table(df))
    return eli_cols
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.25,0.35,0.45,0.55,0.65,.75,.8,.85,.90,.95,1])
    plt.show()
def get_collinear_cols_interdf(df,threshold=0.9,method="pearson",verbose=True):
    """
    function to get the columns of a dataframe with collinear value ratio gt threshold
    only keep one column if there are more than one column within that collinear set
    """
    from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
    columns = list(set(df.columns)-set(["id","user_id","label","target"]))

    df = df[columns]
    if method in ["pearson","kendall","spearman"]:
        corrs = df.corr(method=method)

    elif method == "mi_classif":
        # List of mutual informations
        mis = []
        # Iterate through the columns
        for col in columns:
            # print(col)
            # Calculate correlation with the target
            mi = np.reshape(mutual_info_classif(df.values, df[col].values), -1).tolist()
            # Append the list as a tuple
            mis.append(mi)
        corrs = pd.DataFrame(np.array(mis), columns=df.columns, index=df.columns)
    elif method == "mi_regression":
        # List of mutual informations
        mis = []
        # Iterate through the columns
        for col in df.columns:
            # print(col)
            # Calculate correlation with the target
            mi = np.reshape(mutual_info_regression(df.values, df[col].values), -1).tolist()
            # Append the list as a tuple
            mis.append(mi)
        corrs = pd.DataFrame(np.array(mis), columns=df.columns, index=df.columns)
    if verbose:
        correlation_matrix(corrs)
    # Set the threshold
    threshold = threshold
    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}
    # For each column, record the variables that are above the threshold
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
    # Track columns to remove and columns already examined
    columns_total_lenth = len(columns)
    cols_to_remove = []
    cols_seen = []
    cols_to_remove_pair = []
    # Iterate through columns and correlated columns
    for key, value in above_threshold_vars.items():
        # Keep track of columns already examined
        cols_seen.append(key)
        for x in value:
            if x == key:
                next
            else:
                # Only want to remove one in a pair
                if x not in cols_seen:
                    cols_to_remove.append(x)
                    cols_to_remove_pair.append(key)
    cols_to_remove = list(set(cols_to_remove))
    if verbose:
        print("above threshold colliearn set\n", above_threshold_vars)
        print('Number of columns to remove: {},  out of {} columns : '.format(len(cols_to_remove),columns_total_lenth))
    return cols_to_remove
def get_collinear_cols_intradf(df1,df2,threshold=0.6,method="pearson",verbose=True):
    """
    function to get the corresponding columns in two dataframes with collinear value ratio gt threshold
    """
    from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
    columns1 = list(set(df1.columns)-set(["id","user_id","label","target"]))
    columns2 = list(set(df2.columns)-set(["id","user_id","label","target"]))
    assert set(columns1)==set(columns2),"the columns in the two dataframes must be identical"
    df1 = df1[columns1]
    df2 = df2[columns2]
    cols_to_select = []
    for col in columns1:
        ms = ["pearson", "kendall", "spearman","mi_classif","mi_regression"]
        assert method in ms,"method should in {}".format(ms)
        cor = 0.0
        if method in ["pearson", "kendall", "spearman"]:
            cor = df1[col].corr(df2[col],method=method)
        elif method == "mi_classif":
            cor = np.reshape(mutual_info_classif(df1[[col]].values, df2[col].values), -1).tolist()[0]
        elif method == "mi_regression":
            cor = np.reshape(mutual_info_regression(df1[[col]].values, df2[col].values), -1).tolist()[0]
        if cor>threshold:
            cols_to_select.append(col)
        if verbose:
            print("column {}: {} {}".format(col,method,cor))

    if verbose:
        print('Number of columns gt threshold {}: {},  out of {} columns : '.format(threshold,len(cols_to_select),len(columns1)))
    return cols_to_select
def get_target_related_features(df,target="label",method="mi_classif",threshold=0.0,verbose=True):

    """
    function to get the corresponding columns in two dataframes with collinear value ratio gt threshold
    """
    from sklearn.feature_selection import mutual_info_classif,mutual_info_regression,f_classif,f_regression,chi2
    columns = list(set(df.columns)-set(["id","user_id","label","target"]))
    cols_to_select = []
    cols_of_mi = []
    for col in columns:
        ms = ["pearson", "kendall", "spearman","mi_classif","mi_regression","chi","f_regression","f_classif"]
        assert method in ms,"method should in {}".format(ms)
        cor = 0.0
        if method in ["pearson", "kendall", "spearman"]:
            cor = df[col].corr(df[target],method=method)
        elif method == "mi_classif":
            cor = np.reshape(mutual_info_classif(df[[col]].values, df[target].values), -1).tolist()[0]
        elif method == "mi_regression":
            cor = np.reshape(mutual_info_regression(df[[col]].values, df[target].values), -1).tolist()[0]
        elif method == "f_classif":
            cor = np.reshape(f_classif(df[[col]].values, df[target].values), -1).tolist()[0]
        elif method == "f_regression":
            cor = np.reshape(f_regression(df[[col]].values, df[target].values), -1).tolist()[0]
        elif method == "chi2":
            cor = np.reshape(chi2(df[[col]].values, df[target].values), -1).tolist()[0]
        if cor>threshold:
            cols_to_select.append(col)
        cols_of_mi.append(cor)
        if verbose:
            print("column {}: {} {}".format(col,method,cor))
    feature_importances_mi = sorted(zip(columns,cols_of_mi),
                                      key=lambda x: x[1], reverse=True)
    feature_importances = pd.DataFrame([list(f) for f in feature_importances_mi], columns=["features", "importance"])
    if verbose:
        print('Number of columns gt threshold {}: {},  out of {} columns : '.format(threshold,len(cols_to_select),len(columns)))
        print("feature importance based on {} :".format(method))
        print(feature_importances)
    return cols_to_select,feature_importances

def get_Kbest_feature_lgb(train_x,train_y,val_x,val_y,method="gain",span=(0,1000,1),sorted_feature_list=None,verbose=True):
    """get feature importances by lgb model, supported methods included:
     ["split","gain","shap"]
       "span" is a list with start, end and step index
       "sorted_feature_list" is a feature importance list if provided
    """
    assert span[1]>span[0], "span should be a 3-gram tuple, span[1] > span[0]"
    span = list(span)
    span[1] = min((train_x.shape[1],span[1]))
    score_ls = []
    num_feature_ls = []
    eli_cols = []
    import lightgbm as lgb
    import gc
    params = {
        # "max_bin": 512,
        "learning_rate": 0.01,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "max_depth": -1,
        "verbose": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "subsample_freq": 1,
        "reg_alpha": 0,
        "min_child_weight": 25,
        "random_state": 2018,
        "reg_lambda": 1,
        "n_jobs": -1,
    }
    if sorted_feature_list is None:
        d_train = lgb.Dataset(train_x, label=train_y)
        d_test = lgb.Dataset(val_x, label=val_y)
        print("begin to train ")
        clf_lgb = lgb.train(params, d_train, 4000, valid_sets=[d_train, d_test], early_stopping_rounds=100,
                            verbose_eval=200)
        pre_score_val = clf_lgb.best_score["valid_1"]["auc"]
        pre_score_train = clf_lgb.best_score["training"]["auc"]
        score_ls.append(pre_score_val)
        num_feature_ls.append(span[1]+1)
        if method=="gain":
            feature_importances_gain = sorted(zip(train_x.columns, clf_lgb.feature_importance(importance_type="gain")),
                                              key=lambda x: x[1], reverse=True)
            feature_importances = pd.DataFrame([list(f) for f in feature_importances_gain], columns=["features", "importance"])
        elif method=="split":
            feature_importances_split = sorted(zip(train_x.columns, clf_lgb.feature_importance(importance_type="split")),
                                              key=lambda x: x[1], reverse=True)
            feature_importances = pd.DataFrame([list(f) for f in feature_importances_split], columns=["features", "importance"])
        elif method=="shap":
            import shap
            import numpy as np
            shap.initjs()
            explainer = shap.TreeExplainer(clf_lgb)
            # shap_sample = val_x.sample(frac=1.0)
            shap_sample = train_x.sample(frac=0.6)
            shap_values = explainer.shap_values(shap_sample)
            shap.summary_plot(shap_values, shap_sample, plot_type="bar")
            feature_importances_shap = sorted(zip(train_x.columns, np.mean(np.abs(shap_values), axis=0)),
                                              key=lambda x: x[1], reverse=True)
            feature_importances = pd.DataFrame([list(f) for f in feature_importances_shap],
                                                        columns=["features", "importance"])
        feature_importances.to_csv("../work/feature_importance_eli_cor.csv",header=True,index=False)
        del d_test,d_train,clf_lgb
        gc.collect()
        if verbose:
            print(feature_importances)
            print("feature {} to {}, score {}".format(0,span[1],score_ls[0]))
        num_turn = max((0,int((span[1]-span[0])/span[2])))
        feature_all = feature_importances["features"].unique().tolist()
        for i in range(num_turn):
            print("the {}th turn ".format(i))
            num_feature = span[1]-span[2]*(i+1)
            temp_features = feature_all[0:num_feature]
            d_train_temp = lgb.Dataset(train_x[temp_features], label=train_y)
            d_test_temp = lgb.Dataset(val_x[temp_features], label=val_y)
            print("begin to train ")
            clf_temp = lgb.train(params, d_train_temp, 4000, valid_sets=[d_train_temp, d_test_temp], early_stopping_rounds=100,
                                verbose_eval=200)
            temp_score_val = clf_temp.best_score["valid_1"]["auc"]
            temp_score_train = clf_temp.best_score["training"]["auc"]
            if temp_score_val>pre_score_val and temp_score_train>pre_score_train:
                for f in feature_all[num_feature:num_feature+span[2]]:
                    eli_cols.append(f)
                print("features do not help:",eli_cols)
            pre_score_train = temp_score_train
            pre_score_val = temp_score_val
            score_ls.append(temp_score_val)
            num_feature_ls.append(num_feature)

            del d_test_temp,d_train_temp,clf_temp
        best_score = max(score_ls)
        best_num_feature = num_feature_ls[score_ls.index(best_score)]
        if verbose:
            print("best score {}, best number of feature span {} to {}".format(best_score,0,best_num_feature))
        return feature_all[0:best_num_feature],eli_cols
    else:
        feature_importances = sorted_feature_list
        if verbose:
            print(feature_importances)
        num_turn = max((1,int((span[1]-span[0])/span[2])))
        feature_all = feature_importances["features"].unique().tolist()
        pre_score_val = 0
        pre_score_train = 0
        for i in range(num_turn):
            print("the {}th turn ".format(i))
            num_feature = span[1]-span[2]*i
            temp_features = feature_all[0:num_feature]
            d_train_temp = lgb.Dataset(train_x[temp_features], label=train_y)
            d_test_temp = lgb.Dataset(val_x[temp_features], label=val_y)
            print("begin to train ")
            clf_temp = lgb.train(params, d_train_temp, 4000, valid_sets=[d_train_temp, d_test_temp], early_stopping_rounds=100,
                                verbose_eval=100)
            temp_score_val = clf_temp.best_score["valid_1"]["auc"]
            temp_score_train = clf_temp.best_score["training"]["auc"]
            if i==0:
                pre_score_val = temp_score_val
                pre_score_train = temp_score_train
            if temp_score_val>pre_score_val and temp_score_train>pre_score_train:
                for f in feature_all[num_feature:num_feature+span[2]]:
                    eli_cols.append(f)
                print("features do not help:", eli_cols)
            pre_score_train = temp_score_train
            pre_score_val = temp_score_val
            score_ls.append(temp_score_val)
            num_feature_ls.append(num_feature)
            del d_test_temp,d_train_temp,clf_temp
        best_score = max(score_ls)
        best_num_feature = num_feature_ls[score_ls.index(best_score)]
        if verbose:
            print("best score {}, best number of feature span {} to {}".format(best_score,0,best_num_feature))
        return feature_all[0:best_num_feature],eli_cols


def remove_missing_columns(train, test, threshold=90):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)

    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)

    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])

    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))

    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))

    # Drop the missing columns and return
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)

    return train, test