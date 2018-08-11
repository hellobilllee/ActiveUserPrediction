import  pandas as pd
import numpy as np
import shap

def collinear_columns_to_remove(df,threshold=0.8,method="pearson"):
    from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
    if method in ["pearson","kendall","spearman"]:
        corrs = df.corr(method=method)
    elif method == "mi_classif":
        # List of mutual informations
        mis = []

        # Iterate through the columns
        for col in df.columns:
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
    # Set the threshold
    threshold = threshold

    # Empty dictionary to hold correlated variables
    above_threshold_vars = {}

    # For each column, record the variables that are above the threshold
    for col in corrs:
        above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
    # Track columns to remove and columns already examined
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
    print('Number of columns to remove: ', len(cols_to_remove))
    return cols_to_remove


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