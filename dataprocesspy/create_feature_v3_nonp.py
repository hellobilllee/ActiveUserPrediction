import concurrent
from collections import Counter
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import itertools
import os


def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan
    unique, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])


def ratio_beyond_r_sigma(x, r):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x)) / x.size


def autocorrelation(x, lag):
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.
    if type(x) is pd.Series:
        x = x.values
    if len(x) < lag:
        return np.nan
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x) - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    return sum_product / ((len(x) - lag) * np.var(x))


def binned_entropy(x, max_bins):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)


def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def time_reversal_asymmetry_statistic(x, lag):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((np.roll(x, 2 * -lag) * np.roll(x, 2 * -lag) * np.roll(x, -lag) -
                        np.roll(x, -lag) * x * x)[0:(n - 2 * lag)])


def count_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    m = np.mean(x)
    return np.where(x < m)[0].size


def count_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    m = np.mean(x)
    return np.where(x > m)[0].size


def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x >= np.mean(x))) if x.size > 0 else 0


def longest_strike_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x <= np.mean(x))) if x.size > 0 else 0


def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


def mean_change(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.mean(np.diff(x))


def abs_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


def variance_larger_than_standard_deviation(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    y = np.var(x)
    return 1 if y > np.sqrt(y) else 0


def cid_ce(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    s = np.std(x)
    if s != 0:
        x = (x - np.mean(x)) / s
    else:
        return 0.0
    x = np.diff(x)
    return np.sqrt(np.dot(x, x))


def count_occurence(x, span):
    # if not isinstance(x, (np.ndarray, pd.Series)):
    #     x = np.asarray(x)
    # return np.sum((x >= span[0]) & (x < span[1]))
    count_dict = Counter(list(x))
    # print(count_dict)
    occu = 0
    for i in range(span[0], span[1]):
        if i in count_dict.keys():
            occu += count_dict.get(i)
    return occu


def get_gap(ori, maxSpan, g):
    x = [i for i in list(set(ori)) if i > maxSpan - g]
    if not x:
        return 0
    gap = (len(set(x)) - 1) * 1.0 / (max(x) - min(x)) if len(set(x)) > 1 else min(x) * 1.0 / (maxSpan + min(x))
    return gap


def get_var(lst, maxSpan, v):
    la = [i for i in lst if i > (maxSpan - v)]
    if la:
        return np.var(la)
    else:
        return 0


def get_ratio(lst, e):
    return lst.count(e) * 1.0 / len(lst)


def get_low_std_cols(df, threshold=0.0001, verbose=True):
    """function to get the columns of a dataframe with std lt threshold"""
    columns = list(set(df.columns) - set(["id", "user_id", "label", "target"]))
    eli_cols = []
    for col in columns:
        if df[col].std() < threshold:
            eli_cols.append(col)
    if verbose:
        print("low std colums :", eli_cols)
        print("{} out of {} with std lt threshold {}".format(len(eli_cols), len(columns), threshold))
    return eli_cols


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


def get_missing_ratio_cols(df, threshold=0.5, verbose=True):
    """function to get the columns of a dataframe with missing value ratio gt threshold"""
    columns = list(set(df.columns) - set(["id", "user_id", "label", "target"]))
    eli_cols = []
    df_total_lenth = df.shape[0]
    for col in columns:
        if df[col].isnull().sum() / df_total_lenth > threshold:
            eli_cols.append(col)
    if verbose:
        print("{} out of {} with missing value ratio gt threshold {}".format(len(eli_cols), len(columns), threshold))
        print(missing_values_table(df))
    return eli_cols



def get_workday_rate(x):
    cnt = 0
    for e in list(x):
        if e not in [6, 7, 13, 14, 21, 22, 23, 27, 28]:
            cnt += 1
    return cnt


def get_workday_day(x):
    cnt = 0
    for e in list(set(x)):
        if e not in [6, 7, 13, 14, 21, 22, 23, 27, 28]:
            cnt += 1
    return cnt


def get_weekend_rate(x):
    cnt = 0
    for e in list(x):
        if e in [6, 7, 13, 14, 21, 22, 23, 27, 28]:
            cnt += 1
    return cnt


def get_weekend_day(x):
    cnt = 0
    for e in list(set(x)):
        if e in [6, 7, 13, 14, 21, 22, 23, 27, 28]:
            cnt += 1
    return cnt


def only_in_weekend(x):
    cnt = 1
    weekend = [6, 7, 13, 14, 21, 22, 23, 27, 28]
    workday = [i for i in range(1, 31) if i not in weekend]
    for e in list(set(x)):
        if e in workday:
            cnt = 0
            break
    return cnt


def workday_weekend_rate_ratio(x):
    cnt_work = 0
    cnt_weekend = 0
    weekend = [6, 7, 13, 14, 21, 22, 23, 27, 28]
    workday = [i for i in range(1, 31) if i not in weekend]
    for e in list(x):
        if e in weekend:
            cnt_weekend += 1
        if e in workday:
            cnt_work += 1
    if cnt_work == 0:
        return cnt_weekend * cnt_weekend
    else:
        return cnt_weekend / cnt_work


def workday_weekend_day_ratio(x):
    x = list(set(x))
    cnt_work = 0
    cnt_weekend = 0
    weekend = [6, 7, 13, 14, 21, 22, 23, 27, 28]
    workday = [i for i in range(1, 31) if i not in weekend]
    for e in list(x):
        if e in weekend:
            cnt_weekend += 1
        if e in workday:
            cnt_work += 1
    if cnt_work == 0:
        return cnt_weekend * cnt_weekend
    else:
        return cnt_weekend / cnt_work


def get_mod_var(x):
    x = [i / 7 for i in list(x)]
    return np.var(x)

# ["reg","app","video","act"]

def create_feature(which="reg", trainSpan=(9, 23), base=None):
    import os
    if which == "reg":
        # file_name1 = "../input/fusai/basic_register_log_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
        file_name1 = "../input/fusai/basic_register_log_feature.csv"
        if os.path.exists(file_name1):
            df_user_register_train = pd.read_csv(file_name1, header=0, index_col=None)
            return df_user_register_train
        print("get users from user register log")
        dtype_user_register = {"user_id": np.uint32, "register_day": np.int8, "register_type": np.int8,
                               "device_type": np.uint16}
        df_user_register = pd.read_csv("../input/fusai/user_register_log.txt", header=0, index_col=None,
                                       dtype=dtype_user_register)
        # user_outliers = df_user_register[(df_user_register["device_type"] == 1) | (
        # (df_user_register["register_day"].isin([24, 25, 26])) & (df_user_register["register_type"] == 3) & (
        # (df_user_register["device_type"] == 10566) | (df_user_register["device_type"] == 3036)))][
        #     "user_id"].unique().tolist()
        # df_user_register = df_user_register[~df_user_register["user_id"].isin(user_outliers)]

        df_user_register_train = df_user_register.loc[
            (df_user_register["register_day"] >= 0) & (df_user_register["register_day"] <= 30)]
        del df_user_register
        gc.collect()
        df_user_register_train["register_day_rate"] = (
            df_user_register_train.groupby(by=["register_day"])["register_day"].transform("count")).astype(np.uint16)
        df_user_register_train["register_day_ratio"] = (
            df_user_register_train["register_day_rate"] / len(df_user_register_train)).astype(np.float16)
        df_user_register_train["register_day_type_rate"] = (
            df_user_register_train.groupby(by=["register_day", "register_type"])["register_type"].transform(
                "count")).astype(
            np.uint16)
        df_user_register_train["register_day_type_ratio"] = (
            df_user_register_train["register_day_type_rate"] / df_user_register_train["register_day_rate"]).astype(
            np.float32)
        df_user_register_train["register_day_device_rate"] = (
            df_user_register_train.groupby(by=["register_day", "device_type"])["device_type"].transform(
                "count")).astype(
            np.uint16)
        df_user_register_train["register_day_device_ratio"] = (
            df_user_register_train["register_day_device_rate"] / df_user_register_train["register_day_rate"]).astype(
            np.float32)
        df_user_register_train["register_type_rate"] = (
            df_user_register_train.groupby(by=["register_type"])["register_type"].transform("count")).astype(np.uint16)
        df_user_register_train["register_type_ratio"] = (
            df_user_register_train["register_type_rate"] / len(df_user_register_train)).astype(np.float32)
        df_user_register_train["register_type_device"] = (
            df_user_register_train.groupby(by=["register_type"])["device_type"].transform(
                lambda x: x.nunique())).astype(
            np.uint16)
        df_user_register_train["register_type_device_rate"] = (
            df_user_register_train.groupby(by=["register_type", "device_type"])["device_type"].transform(
                "count")).astype(
            np.uint16)
        df_user_register_train["register_type_device_ratio"] = (
            df_user_register_train["register_type_device_rate"] / df_user_register_train["register_type_rate"]).astype(
            np.float32)
        df_user_register_train["device_type_rate"] = (
            df_user_register_train.groupby(by=["device_type"])["device_type"].transform("count")).astype(np.uint16)
        df_user_register_train["device_type_ratio"] = (
            df_user_register_train["device_type_rate"] / len(df_user_register_train)).astype(np.float32)
        # df_user_register_train["device_type_register"] = (df_user_register_train.groupby(by=["device_type"])["register_type"].transform(lambda x: x.nunique())).astype(np.uint8)
        df_user_register_train["device_type_register_rate"] = (
            df_user_register_train.groupby(by=["device_type", "register_type"])["register_type"].transform(
                "count")).astype(
            np.uint16)
        df_user_register_train["device_type_register_ratio"] = (
            df_user_register_train["device_type_register_rate"] / df_user_register_train["device_type_rate"]).astype(
            np.float32)
        df_user_register_train["register_day_register_type_device_rate"] = (
            df_user_register_train.groupby(by=["register_day", "register_type", "device_type"])[
                "device_type"].transform(
                "count")).astype(np.uint16)
        df_user_register_train["register_day_register_type_device_ratio"] = (
            df_user_register_train["register_day_register_type_device_rate"] / df_user_register_train[
                "register_day_type_rate"]).astype(np.float32)
        df_user_register_train["register_day_device_type_register_rate"] = (
            df_user_register_train.groupby(by=["register_day", "device_type", "register_type"])[
                "register_type"].transform(
                "count")).astype(np.uint16)
        df_user_register_train["register_day_device_type_register_ratio"] = (
            df_user_register_train["register_day_device_type_register_rate"] / df_user_register_train[
                "register_day_device_rate"]).astype(np.float32)

        user_register_feature = ["user_id",
                                 "register_day",
                                 "register_day_ratio",
                                 # "register_day_type_rate",
                                 "register_day_type_ratio",
                                 "register_day_device_ratio",
                                 "register_type_ratio",
                                 "register_type_device",
                                 "register_type_device_ratio",
                                 # "register_day_device_rate",
                                 "device_type_ratio",
                                 "device_type_register_ratio",
                                 "register_day_register_type_device_ratio",
                                 "register_day_device_type_register_ratio"
                                 ]
        # df_user_register_base = df_user_register[["user_id", "register_day"]].drop_duplicates()
        print("begin to drop duplicates for the register feature")
        df_user_register_train = df_user_register_train[user_register_feature].drop_duplicates()
        print("begin to drop the low variance features for the register feature")
        lsc = get_low_std_cols(df_user_register_train)
        df_user_register_train.drop(labels=lsc, axis=1, inplace=True)
        print("begin to drop the columns with missing value ration gt 50%")
        mrc = get_missing_ratio_cols(df_user_register_train)
        df_user_register_train.drop(labels=mrc, axis=1, inplace=True)
        # ds1 = df_user_register_train.describe()
        print(df_user_register_train.describe())
        print("memory usage ", df_user_register_train.memory_usage().sum() / 1024 ** 2)
        print("finish getting basic register log features, get {} features ".format(len(user_register_feature) - 1))
        # ds1.to_csv("kuaishou_stats2.csv", mode='a')
        print("save the basic feature to ../input/fusai/basic_register_log_feature.csv")
        # file_name1 = "../input/fusai/basic_register_log_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
        file_name1 = "../input/fusai/basic_register_log_feature.csv"
        df_user_register_train.to_csv(file_name1, header=True, index=False)
        gc.collect()
        return df_user_register_train
    elif which == "app":
        file_name2 = "../input/fusai/basic_app_launch_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
        if os.path.exists(file_name2):
            df_app_launch_train = pd.read_csv(file_name2, header=0, index_col=None)
            return df_app_launch_train
        print("get users from app launch log")
        dtype_app_launch = {
            "user_id": np.uint32,
            "app_launch_day": np.int8,
        }
        df_app_launch = pd.read_csv("../input/fusai/app_launch_log.txt", header=0, index_col=None,
                                    dtype=dtype_app_launch)
        # df_app_launch = df_app_launch[~df_app_launch["user_id"].isin(user_outliers)]
        df_app_launch = df_app_launch.merge(base, on=["user_id"], how="left").fillna(-1)
        df_app_launch_train = df_app_launch.loc[
            (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]
        # print(df_app_launch_train.describe())
        del df_app_launch
        gc.collect()
        temp_path = "../input/fusai/temp" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + "/"
        print("begin to get the device and type features for app launch log")
        if not os.path.exists(temp_path + "app_launch_temp1.csv"):
            df_app_launch_train_temp = df_app_launch_train[["user_id", "register_type", "device_type"]]
            df_app_launch_train_temp["user_app_launch_register_type_rate"] = \
                (df_app_launch_train_temp.groupby(by=["register_type"])["register_type"].transform("count")).astype(
                    np.uint32)
            df_app_launch_train_temp["user_app_launch_register_type_ratio"] = \
                (df_app_launch_train_temp["user_app_launch_register_type_rate"] / len(df_app_launch_train_temp)).astype(
                    np.float32)
            df_app_launch_train_temp["user_app_launch_device_type_rate"] = \
                (df_app_launch_train_temp.groupby(by=["device_type"])["device_type"].transform("count")).astype(
                    np.uint32)
            df_app_launch_train_temp["user_app_launch_device_type_ratio"] = \
                (df_app_launch_train_temp["user_app_launch_device_type_rate"] / len(df_app_launch_train_temp)).astype(
                    np.float32)
            df_app_launch_train_temp["user_app_launch_register_device_ratio"] = \
                (df_app_launch_train_temp.groupby(by=["register_type", "device_type"])["device_type"].transform(
                    "count") / df_app_launch_train_temp["user_app_launch_register_type_rate"]).astype(np.float32)
            df_app_launch_train_temp["user_app_launch_device_register_ratio"] = \
                (df_app_launch_train_temp.groupby(by=["device_type", "register_type"])["register_type"].transform(
                    "count") / df_app_launch_train_temp["user_app_launch_device_type_rate"]).astype(np.float32)
            app_launch_feature_temp1 = [
                "user_app_launch_register_type_ratio", "user_app_launch_device_type_ratio",
                "user_app_launch_register_device_ratio", "user_app_launch_device_register_ratio"]
            app_launch_temp1_file = temp_path + "app_launch_temp1.csv"
            print("save app_launch_temp1_file to  ", app_launch_temp1_file)
            df_app_launch_train_temp[app_launch_feature_temp1 + ["user_id"]].drop_duplicates().to_csv(
                app_launch_temp1_file, header=True, index=False)
            del df_app_launch_train_temp
            gc.collect()
        df_app_launch_train.drop(
            labels=["register_type", "device_type"], axis=1, inplace=True)
        gc.collect()
        print("begin to get the rate features of each user grouped by app_launch_day")
        df_app_launch_train["user_app_launch_register_time"] = (df_app_launch_train["register_day"].apply(
            lambda x: (trainSpan[1] - x + 1))).astype(np.uint8)
        df_app_launch_train["user_app_launch_register_diff"] = (
            df_app_launch_train["app_launch_day"] - df_app_launch_train["register_day"]).astype(np.uint8)
        df_app_launch_train.drop(
            labels=["register_day"], axis=1, inplace=True)
        gc.collect()

        df_gp_temp_all = df_app_launch_train.groupby(by=["user_id"])
        # df_app_launch_train.drop(labels=["user_app_launch_register_diff"], axis=1,
        #                             inplace=True)
        # gc.collect()
        df_app_launch_train_temp_all = pd.DataFrame()
        df_app_launch_train_temp_all["user_id"] = df_gp_temp_all["app_launch_day"].apply(len).index
        df_app_launch_train_temp_all = df_app_launch_train_temp_all.merge(
            df_app_launch_train[["user_id", "user_app_launch_register_time"]].drop_duplicates(), how="left")

        # del df_app_launch_train
        # gc.collect()
        df_gp_temp = df_gp_temp_all["user_app_launch_register_diff"]
        if not os.path.exists(temp_path + "app_launch_temp2.csv"):
            print("begin to get the timeseries feature of the app launch log(1)")
            df_app_launch_train_temp_all["user_app_launch_max"] = (df_gp_temp.apply(
                lambda x: max(x))).astype(
                np.uint8).values
            df_app_launch_train_temp_all["user_app_launch_min"] = (df_gp_temp.apply(
                lambda x: min(x))).astype(
                np.uint8).values
            df_app_launch_train_temp_all["user_app_launch_mean"] = (df_gp_temp.apply(
                lambda x: int(np.mean(list(x))))).astype(
                np.uint8).values
            df_app_launch_train_temp_all["user_app_launch_median"] = (df_gp_temp.apply(
                lambda x: np.median(list(x)))).astype(
                np.uint8).values
            app_launch_feature_temp2 = [
                "user_app_launch_max", "user_app_launch_min",
                "user_app_launch_mean", "user_app_launch_median"]
            app_launch_temp2_file = temp_path + "app_launch_temp2.csv"
            print("save app_launch_temp2_file to  ", app_launch_temp2_file)
            df_app_launch_train_temp_all[app_launch_feature_temp2 + ["user_id"]].to_csv(
                app_launch_temp2_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp2, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp3.csv"):
            print("begin to get the timeseries feature of the app launch log(2)")
            df_app_launch_train_temp_all["user_app_launch_var_lt_std"] = (df_gp_temp.apply(
                lambda x: variance_larger_than_standard_deviation(x))).astype(np.int8).values
            df_app_launch_train_temp_all["user_app_launch_abs_energy"] = (df_gp_temp.apply(
                lambda x: abs_energy(x))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_cid_ce"] = (df_gp_temp.apply(
                lambda x: cid_ce(x))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_mean_change"] = (df_gp_temp.apply(
                lambda x: mean_change(x))).astype(np.float32).values
            app_launch_feature_temp3 = [
                "user_app_launch_var_lt_std", "user_app_launch_abs_energy",
                "user_app_launch_cid_ce", "user_app_launch_mean_change"]
            app_launch_temp3_file = temp_path + "app_launch_temp3.csv"
            print("save app_launch_temp3_file to  ", app_launch_temp3_file)
            df_app_launch_train_temp_all[app_launch_feature_temp3 + ["user_id"]].to_csv(
                app_launch_temp3_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp3, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp4.csv"):
            print("begin to get the timeseries feature of the app launch log(3)")
            df_app_launch_train_temp_all["user_app_launch_kurtosis"] = (df_gp_temp.apply(
                lambda x: kurtosis(x))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_cam"] = (df_gp_temp.apply(
                lambda x: count_above_mean(x))).astype(np.int8).values
            df_app_launch_train_temp_all["user_app_launch_cbm"] = (df_gp_temp.apply(
                lambda x: count_below_mean(x))).astype(np.int8).values
            df_app_launch_train_temp_all["user_app_launch_tras"] = (df_gp_temp.apply(
                lambda x: time_reversal_asymmetry_statistic(x, 1))).astype(np.float32).values
            app_launch_feature_temp4 = [
                "user_app_launch_kurtosis", "user_app_launch_cam",
                "user_app_launch_cbm", "user_app_launch_tras"]
            app_launch_temp4_file = temp_path + "app_launch_temp4.csv"
            print("save app_launch_temp4_file to  ", app_launch_temp4_file)
            df_app_launch_train_temp_all[app_launch_feature_temp4 + ["user_id"]].to_csv(
                app_launch_temp4_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp4, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp5.csv"):
            print("begin to get the timeseries feature of the app launch log(4)")
            df_app_launch_train_temp_all["user_app_launch_bent"] = (df_gp_temp.apply(
                lambda x: binned_entropy(x, 4))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_autocorr"] = (df_gp_temp.apply(
                lambda x: autocorrelation(x, 1))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_rbrs"] = (df_gp_temp.apply(
                lambda x: ratio_beyond_r_sigma(x, 1))).astype(np.float32).values
            app_launch_feature_temp5 = [
                "user_app_launch_bent", "user_app_launch_autocorr",
                "user_app_launch_rbrs"]
            app_launch_temp5_file = temp_path + "app_launch_temp5.csv"
            print("save app_launch_temp5_file to  ", app_launch_temp5_file)
            df_app_launch_train_temp_all[app_launch_feature_temp5 + ["user_id"]].to_csv(
                app_launch_temp5_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp5, axis=1, inplace=True)
            gc.collect()
        del df_gp_temp
        gc.collect()
        df_gp_temp = df_gp_temp_all["app_launch_day"]
        if not os.path.exists(temp_path + "app_launch_temp6.csv"):
            print("begin to get forward rate features for app launch log(1)")
            app_launch_feature_temp6 = []
            for rf in tqdm(range(0, 3, 1)):
                rate_name_forward = "user_app_launch_rate_f" + str(rf)
                df_app_launch_train_temp_all[rate_name_forward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf + 1)))).astype(
                    np.uint8).values
                rate_name_forward_resInv = rate_name_forward + "_resInv"
                df_app_launch_train_temp_all[rate_name_forward_resInv] = (
                    df_app_launch_train_temp_all[rate_name_forward] * 1.0 / df_app_launch_train_temp_all[
                        "user_app_launch_register_time"]).astype(
                    np.float32)
                app_launch_feature_temp6.append(rate_name_forward)
                app_launch_feature_temp6.append(rate_name_forward_resInv)
            app_launch_temp6_file = temp_path + "app_launch_temp6.csv"
            print("save app_launch_temp6_file to  ", app_launch_temp6_file)
            df_app_launch_train_temp_all[app_launch_feature_temp6 + ["user_id"]].to_csv(
                app_launch_temp6_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp6, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp7.csv"):
            print("begin to get forward rate features for app launch log")
            app_launch_feature_temp7 = []
            for rf in tqdm(range(3, 6, 1)):
                rate_name_forward = "user_app_launch_rate_f" + str(rf)
                df_app_launch_train_temp_all[rate_name_forward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf + 1)))).astype(
                    np.uint8).values
                rate_name_forward_resInv = rate_name_forward + "_resInv"
                df_app_launch_train_temp_all[rate_name_forward_resInv] = (
                    df_app_launch_train_temp_all[rate_name_forward] * 1.0 / df_app_launch_train_temp_all[
                        "user_app_launch_register_time"]).astype(
                    np.float32)
                app_launch_feature_temp7.append(rate_name_forward)
                app_launch_feature_temp7.append(rate_name_forward_resInv)
            app_launch_temp7_file = temp_path + "app_launch_temp7.csv"
            print("save app_launch_temp7_file to  ", app_launch_temp7_file)
            df_app_launch_train_temp_all[app_launch_feature_temp7 + ["user_id"]].to_csv(
                app_launch_temp7_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp7, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp8.csv"):
            print("begin to get backward rate features for app launch log(1)")
            app_launch_feature_temp8 = []
            for rb in tqdm(range(0, 3, 1)):
                rate_name_backward = "user_app_launch_rate_b" + str(rb)
                df_app_launch_train_temp_all[rate_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(
                    np.uint8).values
                rate_name_backward_resInv = rate_name_backward + "_resInv"
                df_app_launch_train_temp_all[rate_name_backward_resInv] = (
                    df_app_launch_train_temp_all[rate_name_backward] * 1.0 / df_app_launch_train_temp_all[
                        "user_app_launch_register_time"]).astype(
                    np.float32)
                app_launch_feature_temp8.append(rate_name_backward)
                app_launch_feature_temp8.append(rate_name_backward_resInv)
            app_launch_temp8_file = temp_path + "app_launch_temp8.csv"
            print("save app_launch_temp8_file to  ", app_launch_temp8_file)
            df_app_launch_train_temp_all[app_launch_feature_temp8 + ["user_id"]].to_csv(
                app_launch_temp8_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp8, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp9.csv"):
            print("begin to get backward rate features for app launch log(2)")
            app_launch_feature_temp9 = []
            for rb in tqdm(range(3, 6, 1)):
                rate_name_backward = "user_app_launch_rate_b" + str(rb)
                df_app_launch_train_temp_all[rate_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(
                    np.uint8).values
                rate_name_backward_resInv = rate_name_backward + "_resInv"
                df_app_launch_train_temp_all[rate_name_backward_resInv] = (
                    df_app_launch_train_temp_all[rate_name_backward] / df_app_launch_train_temp_all[
                        "user_app_launch_register_time"]).astype(
                    np.float32)
                app_launch_feature_temp9.append(rate_name_backward)
                app_launch_feature_temp9.append(rate_name_backward_resInv)
            app_launch_temp9_file = temp_path + "app_launch_temp9.csv"
            print("save app_launch_temp9_file to  ", app_launch_temp9_file)
            df_app_launch_train_temp_all[app_launch_feature_temp9 + ["user_id"]].to_csv(
                app_launch_temp9_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp9, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp10.csv"):
            print("begin to get backward rate features for app launch log(3)")
            app_launch_feature_temp10 = []
            for rb in tqdm(range(6, 9, 1)):
                rate_name_backward = "user_app_launch_rate_b" + str(rb)
                df_app_launch_train_temp_all[rate_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(
                    np.uint8).values
                rate_name_backward_resInv = rate_name_backward + "_resInv"
                df_app_launch_train_temp_all[rate_name_backward_resInv] = (
                    df_app_launch_train_temp_all[rate_name_backward] / df_app_launch_train_temp_all[
                        "user_app_launch_register_time"]).astype(
                    np.float32)
                app_launch_feature_temp10.append(rate_name_backward)
                app_launch_feature_temp10.append(rate_name_backward_resInv)
            app_launch_temp10_file = temp_path + "app_launch_temp10.csv"
            print("save app_launch_temp10_file to  ", app_launch_temp10_file)
            df_app_launch_train_temp_all[app_launch_feature_temp10 + ["user_id"]].to_csv(
                app_launch_temp10_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp10, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp11.csv"):
            print("begin to get backward rate features for app launch log(4)")
            app_launch_feature_temp11 = []
            for rb in tqdm(range(9, 12, 1)):
                rate_name_backward = "user_app_launch_rate_b" + str(rb)
                df_app_launch_train_temp_all[rate_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(
                    np.uint8).values
                rate_name_backward_resInv = rate_name_backward + "_resInv"
                df_app_launch_train_temp_all[rate_name_backward_resInv] = (
                    df_app_launch_train_temp_all[rate_name_backward] / df_app_launch_train_temp_all[
                        "user_app_launch_register_time"]).astype(
                    np.float32)
                app_launch_feature_temp11.append(rate_name_backward)
                app_launch_feature_temp11.append(rate_name_backward_resInv)
            app_launch_temp11_file = temp_path + "app_launch_temp11.csv"
            print("save app_launch_temp11_file to  ", app_launch_temp11_file)
            df_app_launch_train_temp_all[app_launch_feature_temp11 + ["user_id"]].to_csv(
                app_launch_temp11_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp11, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp12.csv"):
            print("get app launch gap feature")
            df_app_launch_train_temp_all["user_app_launch_gap"] = (df_gp_temp.apply(
                lambda x: (len(set(x)) - 1) * 1.0 / (max(x) - min(x)) if len(set(x)) > 1 else min(x) * 1.0 / (
                    trainSpan[1] + min(x)))).astype(np.float32).values
            app_launch_gap_name_ls = ["user_app_launch_gap"]
            for g in tqdm(range(3, 13, 3)):
                gap_name = "user_app_launch_gap_b" + str(g)
                df_app_launch_train_temp_all[gap_name] = (df_gp_temp.apply(
                    lambda x: get_gap(x, trainSpan[1], g))).astype(
                    np.float32).values
                app_launch_gap_name_ls.append(gap_name)
            app_launch_feature_temp12 = app_launch_gap_name_ls
            app_launch_temp12_file = temp_path + "app_launch_temp12.csv"
            print("save app_launch_temp12_file to  ", app_launch_temp12_file)
            df_app_launch_train_temp_all[app_launch_feature_temp12 + ["user_id"]].to_csv(
                app_launch_temp12_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp12, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp13.csv"):
            print("get app launch var feature ")
            df_app_launch_train_temp_all["user_app_launch_var"] = (df_gp_temp.apply(
                lambda x: np.var(list(set(x))))).astype(np.float32).values
            app_launch_var_name_ls = []
            for v in tqdm(range(3, 13, 3)):
                var_name = "user_app_launch_var_b" + str(v)
                df_app_launch_train_temp_all[var_name] = (df_gp_temp.apply(
                    lambda x: get_var(list(set(x)), trainSpan[1], v))).astype(np.float32).values
                app_launch_var_name_ls.append(var_name)
            app_launch_feature_temp13 = [
                                            "user_app_launch_var"] + app_launch_var_name_ls
            app_launch_temp13_file = temp_path + "app_launch_temp13.csv"
            print("save app_launch_temp13_file to  ", app_launch_temp13_file)
            df_app_launch_train_temp_all[app_launch_feature_temp13 + ["user_id"]].to_csv(
                app_launch_temp13_file, header=True, index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp13, axis=1, inplace=True)
            gc.collect()
        del  df_gp_temp
        gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp14.csv"):
            df_gp_temp = df_gp_temp_all["app_launch_day"]
            df_app_launch_train_temp_all["user_app_launch_last_time"] = (df_gp_temp.apply(
                lambda x: trainSpan[1] - max(x))).astype(
                np.uint8).values
            df_app_launch_train_temp_all["user_app_launch_first_time"] = (df_gp_temp.apply(
                lambda x: trainSpan[1] - min(x))).astype(
                np.uint8).values
            app_launch_feature_temp14 = [
                "user_app_launch_register_time",
                "user_app_launch_last_time",
                "user_app_launch_first_time"]
            # df_app_launch_train_temp_all = df_app_launch_train_temp_all[app_launch_feature_temp14]
            app_launch_temp14_file = temp_path + "app_launch_temp14.csv"
            print("save app_launch_temp14_file to  ", app_launch_temp14_file)
            df_app_launch_train_temp_all[app_launch_feature_temp14+["user_id"]].to_csv(app_launch_temp14_file, header=True,
                                                                           index=False)
            df_app_launch_train_temp_all.drop(labels=app_launch_feature_temp14, axis=1, inplace=True)
            del df_gp_temp
            gc.collect()
        if not os.path.exists(temp_path + "app_launch_temp15.csv"):
            df_gp_temp = df_gp_temp_all["app_launch_day"]
            df_app_launch_train_temp_all["user_app_launch_workday_rate"] = (df_gp_temp.apply(
                lambda x: get_workday_rate(x))).astype(np.uint8).values
            df_app_launch_train_temp_all["user_app_launch_weekend_rate"] = (df_gp_temp.apply(
                lambda x: get_weekend_rate(x))).astype(np.uint8).values
            df_app_launch_train_temp_all["user_app_launch_workday_weekend_rate_ratio"] = (df_gp_temp.apply(
                lambda x: workday_weekend_rate_ratio(x))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_only_in_weekend"] = (df_gp_temp.apply(
                lambda x: only_in_weekend(x))).astype(np.float32).values
            df_app_launch_train_temp_all["user_app_launch_mod_var"] = (df_gp_temp.apply(
                lambda x: get_mod_var(x))).astype(np.float32).values
            app_launch_feature_temp15_file = temp_path + "app_launch_temp15.csv"
            app_launch_feature_temp15 = [
                "user_app_launch_workday_rate","user_app_launch_weekend_rate",
                "user_app_launch_workday_weekend_rate_ratio",
                "user_app_launch_only_in_weekend","user_app_launch_mod_var"
            ]
            print("save app_launch_feature_temp15_file to  ", app_launch_feature_temp15_file)
            df_app_launch_train_temp_all[app_launch_feature_temp15+["user_id"]].to_csv(
                app_launch_feature_temp15_file, header=True, index=False)
            del df_gp_temp, df_gp_temp_all, df_app_launch_train
            gc.collect()
        else:
            df_app_launch_train_temp_all = pd.read_csv(temp_path + "app_launch_temp15.csv", header=0, index_col=None)
            print(df_app_launch_train_temp_all.describe())
            del df_gp_temp_all, df_app_launch_train
            gc.collect()
        for i in tqdm(range(1, 15)):
            app_launch_feature_temp_file = temp_path + "app_launch_temp" + str(i) + ".csv"
            df_app_launch_train_temp = pd.read_csv(app_launch_feature_temp_file, header=0, index_col=None)
            print(df_app_launch_train_temp.describe())
            df_app_launch_train_temp_all = df_app_launch_train_temp_all.merge(df_app_launch_train_temp, how="left")
            del df_app_launch_train_temp
            gc.collect()
        print("begin to drop the low variance features for the app_launch feature")
        lsc = get_low_std_cols(df_app_launch_train_temp_all)
        df_app_launch_train_temp_all.drop(labels=lsc, axis=1, inplace=True)
        print("begin to drop the columns with missing value ration gt 50% in app launch log")
        mrc = get_missing_ratio_cols(df_app_launch_train_temp_all)
        df_app_launch_train_temp_all.drop(labels=mrc, axis=1, inplace=True)
        # ds2 = df_app_launch_train_temp_all.describe()
        print(df_app_launch_train_temp_all.describe())
        print("memory usage", df_app_launch_train_temp_all.memory_usage().sum() / 1024 ** 2)
        print("finish getting basic app launch log features, get {} features ".format(
            len(list(df_app_launch_train_temp_all.columns)) - 1))
        # ds1.to_csv("kuaishou_stats2.csv", mode='a')
        file_name2 = "../input/fusai/basic_app_launch_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
        print("save the basic feature to ", file_name2)
        df_app_launch_train_temp_all.to_csv(file_name2, header=True, index=False)
        gc.collect()
        return df_app_launch_train_temp_all
    elif which == "video":
        file_name3 = "../input/fusai/basic_video_create_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
        if os.path.exists(file_name3):
            df_video_create_train = pd.read_csv(file_name3, header=0, index_col=None)
            return df_video_create_train
        print("get users from video create")
        dtype_video_create = {"user_id": np.uint32, "video_create_day": np.int8}
        df_video_create = pd.read_csv("../input/fusai/video_create_log.txt", header=0, index_col=None,
                                      dtype=dtype_video_create)
        # df_video_create = df_video_create[~df_video_create["user_id"].isin(user_outliers)]
        df_video_create = df_video_create.merge(base, on=["user_id"], how="left").fillna(-1)
        df_video_create_train = df_video_create.loc[
            (df_video_create["video_create_day"] >= trainSpan[0]) & (
                df_video_create["video_create_day"] <= trainSpan[1])]
        del df_video_create
        gc.collect()
        temp_path = "../input/fusai/temp" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + "/"
        print("begin to get the device and type features for video create log")
        if not os.path.exists(temp_path + "video_create_temp1.csv"):
            df_video_create_train_temp = df_video_create_train[["user_id", "register_type", "device_type"]]
            df_video_create_train.drop(
                labels=["register_type", "device_type"], axis=1, inplace=True)
            gc.collect()
            df_video_create_train_temp["user_video_create_register_type_rate"] = \
                (df_video_create_train_temp.groupby(by=["register_type"])["register_type"].transform("count")).astype(
                    np.uint32)
            df_video_create_train_temp["user_video_create_register_type_ratio"] = \
                (df_video_create_train_temp["user_video_create_register_type_rate"] / len(
                    df_video_create_train_temp)).astype(np.float32)
            df_video_create_train_temp["user_video_create_device_type_rate"] = \
                (df_video_create_train_temp.groupby(by=["device_type"])["device_type"].transform("count")).astype(
                    np.uint32)
            df_video_create_train_temp["user_video_create_device_type_ratio"] = \
                (df_video_create_train_temp["user_video_create_device_type_rate"] / len(
                    df_video_create_train_temp)).astype(np.float32)
            df_video_create_train_temp["user_video_create_register_device_ratio"] = \
                (df_video_create_train_temp.groupby(by=["register_type", "device_type"])["device_type"].transform(
                    "count") / df_video_create_train_temp["user_video_create_register_type_rate"]).astype(np.float32)
            df_video_create_train_temp["user_video_create_device_register_ratio"] = \
                (df_video_create_train_temp.groupby(by=["device_type", "register_type"])["register_type"].transform(
                    "count") / df_video_create_train_temp["user_video_create_device_type_rate"]).astype(np.float32)
            video_create_feature_temp1 = [
                "user_video_create_register_type_ratio", "user_video_create_device_type_ratio",
                "user_video_create_register_device_ratio", "user_video_create_device_register_ratio"]
            video_create_temp1_file = temp_path + "video_create_temp1.csv"
            print("save video_create_temp1_file to  ", video_create_temp1_file)
            df_video_create_train_temp[video_create_feature_temp1 + ["user_id"]].drop_duplicates().to_csv(
                video_create_temp1_file, header=True, index=False)
            del df_video_create_train_temp
            gc.collect()

        print("begin to get the rate features of each user grouped by video_create_day")
        df_video_create_train["user_video_create_register_time"] = (
            df_video_create_train["register_day"].apply(lambda x: (trainSpan[1] - x + 1))).astype(np.uint8)
        df_video_create_train["user_video_create_register_diff"] = (
            df_video_create_train["video_create_day"] - df_video_create_train["register_day"]).astype(np.int8)
        df_video_create_train.drop(
            labels=["register_day"], axis=1, inplace=True)
        gc.collect()
        df_gp_temp_all = df_video_create_train.groupby(by=["user_id"])
        # df_video_create_train.drop(labels=["user_video_create_register_diff"], axis=1,
        #                             inplace=True)
        # gc.collect()
        df_video_create_train_temp_all = pd.DataFrame()
        df_video_create_train_temp_all["user_id"] = df_gp_temp_all["video_create_day"].apply(len).index
        df_video_create_train_temp_all = df_video_create_train_temp_all.merge(
            df_video_create_train[["user_id", "user_video_create_register_time"]].drop_duplicates(), how="left")
        # del df_video_create_train
        # gc.collect()
        if not os.path.exists(temp_path + "video_create_temp2.csv"):
            print("begin to get the timeseries feature of the video create log pre-drop_dup")
            df_gp_temp = df_gp_temp_all["user_video_create_register_diff"]
            print("begin extract timeseries info")
            df_video_create_train_temp_all["user_video_create_mode"] = (df_gp_temp.apply(
                lambda x: Counter(list(x)).most_common(1)[0][0])).astype(
                np.uint8).values
            df_video_create_train_temp_all["user_video_create_lsbm"] = (df_gp_temp.apply(
                lambda x: longest_strike_below_mean(x))).astype(np.uint8).values
            df_video_create_train_temp_all["user_video_create_lsam"] = (df_gp_temp.apply(
                lambda x: longest_strike_above_mean(x))).astype(np.uint8).values
            df_video_create_train_temp_all["user_video_create_prda"] = (df_gp_temp.apply(
                lambda x: percentage_of_reoccurring_datapoints_to_all_datapoints(x))).astype(np.float32).values
            video_create_feature_temp2 = [
                "user_video_create_mode",
                "user_video_create_lsbm",
                "user_video_create_lsam",
                "user_video_create_prda"
            ]
            video_create_feature_temp2_file = temp_path + "video_create_temp2.csv"
            print("save video_create_feature_temp2_file to  ", video_create_feature_temp2_file)
            df_video_create_train_temp_all[video_create_feature_temp2 + ["user_id"]].to_csv(
                video_create_feature_temp2_file, header=True, index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp2, axis=1, inplace=True)
            gc.collect()
            del df_gp_temp
            gc.collect()

        df_gp_temp = df_gp_temp_all["video_create_day"]
        if not os.path.exists(temp_path + "video_create_temp3.csv"):
            print("begin to get the var  feature of the video create log pre-drop_dup")
            df_video_create_train_temp_all["user_video_create_var"] = (df_gp_temp.apply(
                lambda x: np.var(list(x))) * 1.0).astype(np.float32).values
            video_create_var_name_ls = ["user_video_create_var"]
            print("begin to get the var of the whole video create log")
            for v in tqdm(range(5, 12, 3)):
                var_name = "user_video_create_var_b" + str(v)
                df_video_create_train_temp_all[var_name] = (df_gp_temp.apply(
                    lambda x: get_var(list(x), trainSpan[1], v)) * 1.0).astype(np.float32).values
                video_create_var_name_ls.append(var_name)
            video_create_feature_temp3 = video_create_var_name_ls
            video_create_feature_temp3_file = temp_path + "video_create_temp3.csv"
            print("save video_create_feature_temp3_file to  ", video_create_feature_temp3_file)
            df_video_create_train_temp_all[video_create_feature_temp3 + ["user_id"]].to_csv(
                video_create_feature_temp3_file, header=True, index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp3, axis=1, inplace=True)
            gc.collect()

        for i in tqdm([4, 5, 6]):
            rf = (i - 4) * 2
            print(
                "get the first {} days video create forward feature info ".format(trainSpan[1] - trainSpan[0] - rf))
            video_create_feature_temprf_file = temp_path + "video_create_temp" + str(int(4 + rf / 2)) + ".csv"
            if not os.path.exists(video_create_feature_temprf_file):
                rate_name_forward = "user_video_create_rate_f" + str(rf)
                rate_name_forward_resInv = rate_name_forward + "_resInv"
                day_name_forward = "user_video_create_day_f" + str(rf)
                day_name_forward_resInv = day_name_forward + "_resInv"
                frequency_name_forward = "user_video_create_frequency_f" + str(rf)
                df_video_create_train_temp_all[rate_name_forward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf + 1)))).astype(np.uint16).values
                df_video_create_train_temp_all[day_name_forward] = (df_gp_temp.apply(
                    lambda x: count_occurence(set(x), (trainSpan[0], trainSpan[1] - rf + 1)))).astype(
                    np.uint8).values
                df_video_create_train_temp_all[frequency_name_forward] = (
                    df_video_create_train_temp_all[rate_name_forward] / (df_video_create_train_temp_all[
                                                                             day_name_forward] + 0.00000001)).astype(
                    np.float32)
                df_video_create_train_temp_all[rate_name_forward_resInv] = (
                    df_video_create_train_temp_all[rate_name_forward] / \
                    df_video_create_train_temp_all["user_video_create_register_time"]).astype(
                    np.float32)
                df_video_create_train_temp_all[day_name_forward_resInv] = (
                df_video_create_train_temp_all[day_name_forward] / \
                df_video_create_train_temp_all["user_video_create_register_time"]).astype(
                    np.float32)
                video_create_feature_temprf = [rate_name_forward, day_name_forward, frequency_name_forward,
                                               day_name_forward_resInv, rate_name_forward_resInv]
                print(
                    "save video_create_feature_temprf{}_file to {}".format(rf, video_create_feature_temprf_file))
                df_video_create_train_temp_all[video_create_feature_temprf + ["user_id"]].to_csv(
                    video_create_feature_temprf_file, header=True, index=False)
                df_video_create_train_temp_all.drop(labels=video_create_feature_temprf, axis=1, inplace=True)
                gc.collect()
        for i in tqdm([7, 8, 9, 10, 11]):
            rb = (i - 7) * 3
            print("get the last {} days video create backward feature info ".format(rb))
            video_create_feature_temprb_file = temp_path + "video_create_temp" + str(int(7 + rb / 3)) + ".csv"
            video_create_feature_temprb = []
            if not os.path.exists(video_create_feature_temprb_file):
                rate_name_backward = "user_video_create_rate_b" + str(rb)
                rate_name_backward_resInv = rate_name_backward + "_resInv"
                day_name_backward = "user_video_create_day_b" + str(rb)
                day_name_backward_resInv = day_name_backward + "_resInv"
                video_create_feature_temprb.append(rate_name_backward)
                video_create_feature_temprb.append(rate_name_backward_resInv)
                video_create_feature_temprb.append(day_name_backward)
                video_create_feature_temprb.append(day_name_backward_resInv)
                df_video_create_train_temp_all[rate_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint16).values
                df_video_create_train_temp_all[rate_name_backward_resInv] = (
                    df_video_create_train_temp_all[rate_name_backward] / \
                    df_video_create_train_temp_all[
                        "user_video_create_register_time"]).astype(
                    np.float32)
                df_video_create_train_temp_all[day_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(set(x), (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(
                    np.uint8).values
                df_video_create_train_temp_all[day_name_backward_resInv] = (
                    df_video_create_train_temp_all[day_name_backward] / \
                    df_video_create_train_temp_all[
                        "user_video_create_register_time"]).astype(
                    np.float32)
                if rb != 0:
                    frequency_name_backward = "user_video_create_frequency_b" + str(rb)
                    video_create_feature_temprb.append(frequency_name_backward)
                    df_video_create_train_temp_all[frequency_name_backward] = (
                        df_video_create_train_temp_all[rate_name_backward] / (df_video_create_train_temp_all[
                                                                                  day_name_backward] + 0.00000001)).astype(
                        np.float32)
                print("save video_create_feature_temprb{}_file to {}".format(rb, video_create_feature_temprb_file))
                df_video_create_train_temp_all[video_create_feature_temprb + ["user_id"]].to_csv(
                    video_create_feature_temprb_file, header=True, index=False)
                df_video_create_train_temp_all.drop(labels=video_create_feature_temprb, axis=1, inplace=True)
                gc.collect()
        if not os.path.exists(temp_path + "video_create_temp12.csv"):
            print("begin to get the video create gap")
            df_video_create_train_temp_all["user_video_create_gap"] = (df_gp_temp.apply(
                lambda x: (len(set(x)) - 1) * 1.0 / (max(x) - min(x)) if len(set(x)) > 1 else min(x) * 1.0 / (
                    trainSpan[1] + min(x))) * 1.0).astype(np.float32).values
            video_create_gap_name_ls = ["user_video_create_gap"]
            for g in tqdm(range(5, 12, 3)):
                gap_name = "user_video_create_gap_b" + str(g)
                df_video_create_train_temp_all[gap_name] = (df_gp_temp.apply(
                    lambda x: get_gap(x, trainSpan[1], g)) * 1.0).astype(np.float32).values
                video_create_gap_name_ls.append(gap_name)

            video_create_feature_temp12 = video_create_gap_name_ls
            video_create_feature_temp12_file = temp_path + "video_create_temp12.csv"
            print("save video_create_feature_temp12_file to ", video_create_feature_temp12_file)
            df_video_create_train_temp_all[video_create_feature_temp12 + ["user_id"]].to_csv(
                video_create_feature_temp12_file, header=True,
                index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp12, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "video_create_temp13.csv"):
            print("begin to get the video create day var")
            df_video_create_train_temp_all["user_video_create_day_var"] = (df_gp_temp.apply(
                lambda x: np.var(list(set(x)))) * 1.0).astype(np.float32).values
            video_create_day_var_name_ls = ["user_video_create_day_var"]
            for v in tqdm(range(5, 12, 3)):
                day_var_name = "user_video_create_day_var_b" + str(v)
                df_video_create_train_temp_all[day_var_name] = (df_gp_temp.apply(
                    lambda x: get_var(list(set(x)), trainSpan[1], v)) * 1.0).astype(np.float32).values
                video_create_day_var_name_ls.append(day_var_name)
            video_create_feature_temp13 = video_create_day_var_name_ls
            video_create_feature_temp13_file = temp_path + "video_create_temp13.csv"
            print("save video_create_feature_temp13_file to ", video_create_feature_temp13_file)
            df_video_create_train_temp_all[video_create_feature_temp13 + ["user_id"]].to_csv(
                video_create_feature_temp13_file, header=True, index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp13, axis=1, inplace=True)
            gc.collect()
        del df_gp_temp
        gc.collect()
        df_gp_temp = df_gp_temp_all["user_video_create_register_diff"]
        # df_gp_temp = df_gp_temp_all["user_video_create_register_diff"]
        if not os.path.exists(temp_path + "video_create_temp14.csv"):
            print("get the timeseries feature of the user activity log after drop_dup(1)")
            df_video_create_train_temp_all["user_video_create_max"] = (df_gp_temp.apply(
                lambda x: max(x))).astype(
                np.uint8).values
            df_video_create_train_temp_all["user_video_create_min"] = (df_gp_temp.apply(
                lambda x: min(x))).astype(
                np.uint8).values
            df_video_create_train_temp_all["user_video_create_mean"] = (df_gp_temp.apply(
                lambda x: int(np.mean(list(set(x)))))).astype(
                np.uint8).values
            df_video_create_train_temp_all["user_video_create_median"] = (df_gp_temp.apply(
                lambda x: np.median(list(set(x))))).astype(
                np.uint8).values
            video_create_feature_temp14 = ["user_video_create_max", "user_video_create_min",
                                           "user_video_create_mean", "user_video_create_median"]
            video_create_feature_temp14_file = temp_path + "video_create_temp14.csv"
            print("save video_create_feature_temp14_file to ", video_create_feature_temp14_file)
            df_video_create_train_temp_all[video_create_feature_temp14 + ["user_id"]].to_csv(
                video_create_feature_temp14_file, header=True,
                index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp14, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "video_create_temp15.csv"):
            print("get the timeseries feature of the user activity log after drop_dup(2)")
            df_video_create_train_temp_all["user_video_create_var_lt_std"] = (df_gp_temp.apply(
                lambda x: variance_larger_than_standard_deviation(x))).astype(np.int8).values
            df_video_create_train_temp_all["user_video_create_tras"] = (df_gp_temp.apply(
                lambda x: time_reversal_asymmetry_statistic(x, 1))).astype(np.float32).values
            df_video_create_train_temp_all["user_video_create_abs_energy"] = (df_gp_temp.apply(
                lambda x: abs_energy(x))).astype(np.float32).values
            df_video_create_train_temp_all["user_video_create_cid_ce"] = (df_gp_temp.apply(
                lambda x: cid_ce(list(set(x))))).astype(np.float32).values
            video_create_feature_temp15 = ["user_video_create_abs_energy", "user_video_create_cid_ce",
                                           "user_video_create_var_lt_std", "user_video_create_tras"]
            video_create_feature_temp15_file = temp_path + "video_create_temp15.csv"
            print("save video_create_feature_temp15_file to ", video_create_feature_temp15_file)
            df_video_create_train_temp_all[video_create_feature_temp15 + ["user_id"]].to_csv(
                video_create_feature_temp15_file, header=True,
                index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp15, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "video_create_temp16.csv"):
            print("get the timeseries feature of the user activity log after drop_dup(3)")
            df_video_create_train_temp_all["user_video_create_mean_change"] = (df_gp_temp.apply(
                lambda x: mean_change(list(set(x))))).astype(np.float32).values
            df_video_create_train_temp_all["user_video_create_kurtosis"] = (df_gp_temp.apply(
                lambda x: kurtosis(list(set(x))))).astype(np.float32).values
            df_video_create_train_temp_all["user_video_create_cam"] = (df_gp_temp.apply(
                lambda x: count_above_mean(list(set(x))))).astype(np.int8).values
            df_video_create_train_temp_all["user_video_create_cbm"] = (df_gp_temp.apply(
                lambda x: count_below_mean(list(set(x))))).astype(np.int8).values
            video_create_feature_temp16 = [
                "user_video_create_mean_change", "user_video_create_kurtosis",
                "user_video_create_cam", "user_video_create_cbm"]
            video_create_feature_temp16_file = temp_path + "video_create_temp16.csv"
            print("save video_create_feature_temp16_file to ", video_create_feature_temp16_file)
            df_video_create_train_temp_all[video_create_feature_temp16 + ["user_id"]] \
                .to_csv(video_create_feature_temp16_file, header=True,
                        index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp16, axis=1, inplace=True)
            gc.collect()
        if not os.path.exists(temp_path + "video_create_temp17.csv"):
            print("get the timeseries feature of the user activity log after drop_dup(3)")
            df_video_create_train_temp_all["user_video_create_bent"] = (df_gp_temp.apply(
                lambda x: binned_entropy(x, 4))).astype(np.float32).values
            df_video_create_train_temp_all["user_video_create_autocorr"] = (df_gp_temp.apply(
                lambda x: autocorrelation(x, 1))).astype(np.float32).values
            df_video_create_train_temp_all["user_video_create_rbrs"] = (df_gp_temp.apply(
                lambda x: ratio_beyond_r_sigma(x, 1))).astype(np.float32).values
            video_create_feature_temp17 = [
                "user_video_create_bent", "user_video_create_autocorr", "user_video_create_rbrs"]
            video_create_feature_temp17_file = temp_path + "video_create_temp17.csv"
            print("save video_create_feature_temp17_file to ", video_create_feature_temp17_file)
            df_video_create_train_temp_all[video_create_feature_temp17 + ["user_id"]].to_csv(
                video_create_feature_temp17_file, header=True,
                index=False)
            df_video_create_train_temp_all.drop(labels=video_create_feature_temp17, axis=1, inplace=True)
            gc.collect()
        del df_gp_temp
        gc.collect()

        if not os.path.exists(temp_path + "video_create_temp18.csv"):
            df_gp_temp = df_gp_temp_all["video_create_day"]
            df_video_create_train_temp_all["user_video_create_last_time"] = (df_gp_temp.apply(
                lambda x: trainSpan[1] - max(x))).astype(np.uint8).values
            df_video_create_train_temp_all["user_video_create_first_time"] = (df_gp_temp.apply(
                lambda x: trainSpan[1] - min(x))).astype(np.uint8).values
            del df_gp_temp, df_gp_temp_all, df_video_create_train
            gc.collect()
            video_create_feature_temp18 = ["user_id",
                                           "user_video_create_register_time",
                                           "user_video_create_last_time",
                                           "user_video_create_first_time"
                                           ]
            print(df_video_create_train_temp_all.describe())
            video_create_feature_temp18_file = temp_path + "video_create_temp18.csv"
            print("save video_create_feature_temp18_file to ", video_create_feature_temp18_file)
            df_video_create_train_temp_all[video_create_feature_temp18].to_csv(video_create_feature_temp18_file,
                                                                            header=True, index=False)
        else:
            df_video_create_train_temp_all = pd.read_csv(temp_path + "video_create_temp18.csv", header=0,
                                                         index_col=None)
            print(df_video_create_train_temp_all.describe())
            del df_gp_temp_all, df_video_create_train
            gc.collect()
        for i in tqdm(range(1, 18)):
            video_create_feature_temp_file = temp_path + "video_create_temp" + str(i) + ".csv"
            df_video_create_train_temp = pd.read_csv(video_create_feature_temp_file, header=0, index_col=None)
            print(df_video_create_train_temp.describe())
            df_video_create_train_temp_all = df_video_create_train_temp_all.merge(df_video_create_train_temp,
                                                                                  how="left")
            del df_video_create_train_temp
            gc.collect()
        print("begin to drop the low variance features for the video create feature")
        lsc = get_low_std_cols(df_video_create_train_temp_all)
        df_video_create_train_temp_all.drop(labels=lsc, axis=1, inplace=True)
        print("begin to drop the columns with missing value ration gt 50% in video_create log")
        mrc = get_missing_ratio_cols(df_video_create_train_temp_all)
        df_video_create_train_temp_all.drop(labels=mrc, axis=1, inplace=True)

        print(df_video_create_train_temp_all.describe())
        # ds3.to_csv("kuaishou_stats2.csv", mode='a')
        print("memory usage", df_video_create_train_temp_all.memory_usage().sum() / 1024 ** 2)
        print("finish getting basic video creat log features, get {} features ".format(
            len(list(df_video_create_train_temp_all.columns)) - 1))
        # ds1.to_csv("kuaishou_stats2.csv", mode='a')
        file_name3 = "../input/fusai/basic_video_create_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
        print("save the basic feature to ", file_name3)
        df_video_create_train_temp_all.to_csv(file_name3, header=True, index=False)
        gc.collect()
        return df_video_create_train_temp_all
    elif which == "act":
        file_name4 = "../input/fusai/basic_user_activity_feature_" + str(trainSpan[0]) + "_" + str(
            trainSpan[1]) + ".csv"
        if os.path.exists(file_name4):
            df_user_activity_train = pd.read_csv(file_name4, header=0, index_col=None)
            return df_user_activity_train
        temp_path = "../input/fusai/temp" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + "/"
        print("get users from user activity log")
        dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.int8, "page": np.int8,
                               "video_id": np.uint32,
                               "author_id": np.uint32, "action_type": np.int8}
        df_user_activity = pd.read_csv("../input/fusai/user_activity_log.txt", header=0, index_col=None,
                                       dtype=dtype_user_activity)
        # df_user_activity = df_user_activity[~df_user_activity["user_id"].isin(user_outliers)]
        df_user_activity = df_user_activity.merge(base, on=["user_id"], how="left").fillna(-1)

        df_user_activity_train = df_user_activity.loc[
            (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
                df_user_activity["user_activity_day"] <= trainSpan[1])]
        del df_user_activity
        gc.collect()
        print("begin to get user activity features")

        print("begin to get the register and device type features for the activity log")
        if not os.path.exists(temp_path + "user_activity_temp1.csv"):
            df_user_activity_train_temp = df_user_activity_train[["user_id", "register_type", "device_type"]]
            df_user_activity_train_temp["user_user_activity_register_type_rate"] = \
                (df_user_activity_train_temp.groupby(by=["register_type"])["register_type"].transform("count")).astype(
                    np.uint32)
            df_user_activity_train_temp["user_user_activity_register_type_ratio"] = \
                (df_user_activity_train_temp["user_user_activity_register_type_rate"] / len(
                    df_user_activity_train_temp)).astype(np.float32)
            df_user_activity_train_temp["user_user_activity_device_type_rate"] = \
                (df_user_activity_train_temp.groupby(by=["device_type"])["device_type"].transform("count")).astype(
                    np.uint32)
            df_user_activity_train_temp["user_user_activity_device_type_ratio"] = \
                (df_user_activity_train_temp["user_user_activity_device_type_rate"] / len(
                    df_user_activity_train_temp)).astype(np.float32)
            df_user_activity_train_temp["user_user_activity_register_device_ratio"] = \
                (df_user_activity_train_temp.groupby(by=["register_type", "device_type"])["device_type"].transform(
                    "count") / df_user_activity_train_temp["user_user_activity_register_type_rate"]).astype(np.float32)
            df_user_activity_train_temp["user_user_activity_device_register_ratio"] = \
                (df_user_activity_train_temp.groupby(by=["device_type", "register_type"])["register_type"].transform(
                    "count") / df_user_activity_train_temp["user_user_activity_device_type_rate"]).astype(np.float32)
            user_activity_feature_temp1 = ["user_id",
                                           "user_user_activity_register_type_ratio",
                                           "user_user_activity_device_type_ratio",
                                           "user_user_activity_register_device_ratio",
                                           "user_user_activity_device_register_ratio"]
            user_activity_temp1_file = temp_path + "user_activity_temp1.csv"
            print("save user_activity_temp1_file to  ", user_activity_temp1_file)
            df_user_activity_train_temp[user_activity_feature_temp1].drop_duplicates().to_csv(
                user_activity_temp1_file, header=True, index=False)
            del df_user_activity_train_temp
            gc.collect()
        df_user_activity_train.drop(
            labels=["register_type", "device_type"], axis=1, inplace=True)
        gc.collect()

        print("begin to get features for activity log")
        df_user_activity_train["user_user_activity_register_time"] = (
            df_user_activity_train["register_day"].apply(lambda x: (trainSpan[1] - x + 1))).astype(np.uint8)
        df_user_activity_train["user_user_activity_register_diff"] = (
            df_user_activity_train["user_activity_day"] - df_user_activity_train["register_day"]).astype(np.int8)
        df_user_activity_train.drop(
            labels=["register_day"], axis=1, inplace=True)
        gc.collect()

        df_gp_temp_all = df_user_activity_train.groupby(by=["user_id"])
        # df_user_activity_train.drop(labels=["user_user_activity_register_diff","page", "action_type"],axis=1,inplace=True)
        # gc.collect()
        df_user_activity_train_temp_all = pd.DataFrame()
        df_user_activity_train_temp_all["user_id"] = df_gp_temp_all["user_activity_day"].apply(len).index
        df_user_activity_train_temp_all = df_user_activity_train_temp_all.merge(
            df_user_activity_train[["user_id", "user_user_activity_register_time"]].drop_duplicates(), how="left")
        # del df_user_activity_train
        # gc.collect()

        if not os.path.exists(temp_path + "user_activity_temp2.csv"):
            print("begin to get author id in user id rate")
            user_activity_author = df_user_activity_train["author_id"].unique().tolist()
            user_authors = df_user_activity_train.loc[
                df_user_activity_train["user_id"].isin(user_activity_author), "user_id"].tolist()
            user_author_count_dict = Counter(list(user_authors))
            df_user_activity_train["user_in_author_rate"] = df_user_activity_train["user_id"].apply(
                lambda x: 0 if x not in user_author_count_dict.keys() else user_author_count_dict.get(x)).astype(
                np.uint16)
            df_user_activity_train_temp_all = df_user_activity_train_temp_all.merge(
                df_user_activity_train[["user_id", "user_in_author_rate"]].drop_duplicates(), how="left")
            del user_activity_author, user_authors, user_author_count_dict
            print("begin to extract video info ")
            df_user_activity_train_temp_all["user_video_rate"] = (df_gp_temp_all["video_id"].apply(
                lambda x: x.nunique())).astype(np.uint16).values
            df_user_activity_train_temp_all["user_video_rate_resInv"] = (
                df_user_activity_train_temp_all["user_video_rate"] / df_user_activity_train_temp_all[
                    "user_user_activity_register_time"]).astype(np.float32)
            print("begin to extract author info")
            df_user_activity_train_temp_all["user_author_rate"] = (df_gp_temp_all["author_id"].apply(
                lambda x: x.nunique())).astype(np.uint16).values
            df_user_activity_train_temp_all["user_author_rate_resInv"] = (
                df_user_activity_train_temp_all["user_author_rate"] / df_user_activity_train_temp_all[
                    "user_user_activity_register_time"]).astype(np.float32)
            user_activity_feature_temp2_file = temp_path + "user_activity_temp2.csv"
            user_activity_feature_temp2 = ["user_in_author_rate", "user_video_rate", "user_video_rate_resInv",
                                           "user_author_rate", "user_author_rate_resInv"]
            print("save user_activity_feature_temp2_file to  ", user_activity_feature_temp2_file)
            df_user_activity_train_temp_all[user_activity_feature_temp2 + ["user_id"]].to_csv(
                user_activity_feature_temp2_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp2, axis=1, inplace=True)
            gc.collect()
        df_user_activity_train.drop(labels=["video_id", "author_id"], axis=1, inplace=True)
        gc.collect()

        print("begin to get the page action info")
        for i in tqdm([0,1,2,3,4]):
            user_activity_feature_temp_file = temp_path + "user_activity_temp" + str(i + 3) + ".csv"
            user_activity_feature_temp = []
            if not os.path.exists(user_activity_feature_temp_file):
                df_gp_temp1 = df_gp_temp_all["page"]
                df_gp_temp2 = df_gp_temp_all["page", "action_type"]
                page_ratio_name = "user_page" + str(i) + "_ratio"
                df_user_activity_train_temp_all[page_ratio_name] = df_gp_temp1.apply(
                    lambda x: list(x).count(i) / (len(x) + 0.00000001)).astype(np.float32).values
                user_activity_feature_temp.append(page_ratio_name)
                for j in tqdm(range(0, 6)):
                    page_action_ratio_name = "user_page" + str(i) + "_action" + str(j) + "_ratio"
                    user_activity_feature_temp.append(page_action_ratio_name)
                    df_user_activity_train_temp_all[page_action_ratio_name] = df_gp_temp2.apply(
                        lambda x: len(x.loc[(x["page"] == i) & (x["action_type"] == j)]) / (
                        len(x.loc[(x["page"] == i)]) + 0.00000001)).astype(np.float32).values
                print("save user activity page{} feature to {} ".format(i, user_activity_feature_temp_file))
                df_user_activity_train_temp_all[user_activity_feature_temp + ["user_id"]].to_csv(
                    user_activity_feature_temp_file, header=True, index=False)
                df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp, axis=1, inplace=True)
                del df_gp_temp1, df_gp_temp2
                gc.collect()
        print("begin to get the action page info")
        for i in tqdm([0,1,2,3,4,5]):
            user_activity_feature_temp_file = temp_path + "user_activity_temp" + str(i + 8) + ".csv"
            user_activity_feature_temp = []
            if not os.path.exists(user_activity_feature_temp_file):
                df_gp_temp1 = df_gp_temp_all["page"]
                df_gp_temp2 = df_gp_temp_all["page", "action_type"]
                action_type_ratio_name = "user_action_type" + str(i) + "_ratio"
                user_activity_feature_temp.append(action_type_ratio_name)
                df_user_activity_train_temp_all[action_type_ratio_name] = df_gp_temp1.apply(
                    lambda x: list(x).count(i) / (len(x) + 0.00000001)).astype(np.float32).values
                for j in tqdm(range(0, 5)):
                    action_page_ratio_name = "user_action" + str(i) + "_page" + str(j) + "_ratio"
                    user_activity_feature_temp.append(action_page_ratio_name)
                    df_user_activity_train_temp_all[action_page_ratio_name] = df_gp_temp2.apply(
                        lambda x: len(x.loc[(x["page"] == j) & (x["action_type"] == i)]) / (len(
                            x.loc[(x["action_type"] == i)]) + 0.00000001)).astype(np.float32).values
                print("save user activity action{} feature to {} ".format(i, user_activity_feature_temp_file))
                df_user_activity_train_temp_all[user_activity_feature_temp + ["user_id"]].to_csv(
                    user_activity_feature_temp_file, header=True, index=False)
                df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp, axis=1, inplace=True)
                del df_gp_temp1, df_gp_temp2
                gc.collect()
        df_user_activity_train.drop(labels=["page", "action_type"], axis=1, inplace=True)
        gc.collect()
        print("begin to get time series and var features ")
        if not os.path.exists(temp_path + "user_activity_temp14.csv"):
            print("begin to get the timeseries feature of the user activity log pre-drop_dup")
            df_gp_temp = df_gp_temp_all["user_user_activity_register_diff"]
            # print("begin extract timeseries info")
            df_user_activity_train_temp_all["user_user_activity_mode"] = (df_gp_temp.apply(
                lambda x: Counter(list(x)).most_common(1)[0][0])).astype(
                np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_lsbm"] = (df_gp_temp.apply(
                lambda x: longest_strike_below_mean(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_lsam"] = (df_gp_temp.apply(
                lambda x: longest_strike_above_mean(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_prda"] = (df_gp_temp.apply(
                lambda x: percentage_of_reoccurring_datapoints_to_all_datapoints(x))).astype(np.float32).values
            user_activity_feature_temp14_file = temp_path + "user_activity_temp14.csv"
            print("save user_activity_feature_temp14_file to  ", user_activity_feature_temp14_file)
            user_activity_feature_temp14 = [
                "user_user_activity_mode", "user_user_activity_lsbm",
                "user_user_activity_lsam", "user_user_activity_prda"]
            df_user_activity_train_temp_all[user_activity_feature_temp14 + ["user_id"]].to_csv(
                user_activity_feature_temp14_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp14, axis=1, inplace=True)
            del df_gp_temp
            gc.collect()
        print("begin to get the var of the whole activity")
        if not os.path.exists(temp_path + "user_activity_temp15.csv"):
            df_gp_temp = df_gp_temp_all["user_activity_day"]
            df_user_activity_train_temp_all["user_user_activity_var"] = (df_gp_temp.apply(
                lambda x: np.var(list(x)))).astype(np.float32).values
            user_activity_var_name_ls = ["user_user_activity_var"]
            for v in tqdm(range(3, 13, 3)):
                var_name = "user_user_activity_var_b" + str(v)
                df_user_activity_train_temp_all[var_name] = (df_gp_temp.apply(
                    lambda x: get_var(list(x), trainSpan[1], v))).astype(np.float32).values
                user_activity_var_name_ls.append(var_name)
            user_activity_feature_temp15_file = temp_path + "user_activity_temp15.csv"
            user_activity_feature_temp15 = user_activity_var_name_ls
            print("save user_activity_feature_temp15_file to  ", user_activity_feature_temp15_file)
            df_user_activity_train_temp_all[user_activity_feature_temp15 + ["user_id"]].to_csv(
                user_activity_feature_temp15_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp15, axis=1, inplace=True)
            gc.collect()
            del df_gp_temp
            gc.collect()

        print("begin to get the farward and backward rate features")
        for i in tqdm(range(0,6)):
            rf = i
            print("get the first {} days activity info ".format(trainSpan[1] - trainSpan[0] - rf + 1))
            user_activity_feature_temprf_file = temp_path + "user_activity_temp" + str(16 + rf) + ".csv"
            if not os.path.exists(user_activity_feature_temprf_file):
                df_gp_temp = df_gp_temp_all["user_activity_day"]
                rate_name_forward = "user_user_activity_rate_f" + str(rf)
                rate_name_forward_resInv = rate_name_forward + "_resInv"
                day_name_forward = "user_user_activity_day_f" + str(rf)
                day_name_forward_resInv = day_name_forward + "_resInv"
                frequency_name_forward = "user_user_activity_frequency_f" + str(rf)

                df_user_activity_train_temp_all[rate_name_forward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf + 1)))).astype(np.uint16).values
                df_user_activity_train_temp_all[day_name_forward] = (df_gp_temp.apply(
                    lambda x: count_occurence(set(x), (trainSpan[0], trainSpan[1] - rf + 1)))).astype(np.uint8).values
                df_user_activity_train_temp_all[frequency_name_forward] = (
                    df_user_activity_train_temp_all[rate_name_forward] / (
                    df_user_activity_train_temp_all[day_name_forward] + 0.00000001)).astype(
                    np.float32)

                df_user_activity_train_temp_all[rate_name_forward_resInv] = (
                df_user_activity_train_temp_all[rate_name_forward] / \
                df_user_activity_train_temp_all[
                    "user_user_activity_register_time"]).astype(
                    np.float32)

                df_user_activity_train_temp_all[day_name_forward_resInv] = (
                df_user_activity_train_temp_all[day_name_forward] / \
                df_user_activity_train_temp_all[
                    "user_user_activity_register_time"]).astype(
                    np.float32)

                user_activity_feature_temprf = [rate_name_forward, day_name_forward, frequency_name_forward,
                                                day_name_forward_resInv, rate_name_forward_resInv]
                print("save user_activity_feature_temprf{}_file to {}".format(rf, user_activity_feature_temprf_file))
                df_user_activity_train_temp_all[user_activity_feature_temprf + ["user_id"]].to_csv(
                    user_activity_feature_temprf_file, header=True, index=False)
                df_user_activity_train_temp_all.drop(labels=user_activity_feature_temprf, axis=1, inplace=True)
                gc.collect()
                del  df_gp_temp
                gc.collect()
        for i in tqdm(range(0,12)):
            rb = i
            print("get the last {} days activity info ".format(rb))
            user_activity_feature_temprb_file = temp_path + "user_activity_temp" + str(22 + rb) + ".csv"
            user_activity_feature_temprb = []
            if not os.path.exists(user_activity_feature_temprb_file):
                df_gp_temp = df_gp_temp_all["user_activity_day"]
                rate_name_backward = "user_user_activity_rate_b" + str(rb)
                rate_name_backward_resInv = rate_name_backward + "_resInv"
                day_name_backward = "user_user_activity_day_b" + str(rb)
                day_name_backward_resInv = day_name_backward + "_resInv"
                user_activity_feature_temprb.append(rate_name_backward)
                user_activity_feature_temprb.append(rate_name_backward_resInv)
                user_activity_feature_temprb.append(day_name_backward)
                user_activity_feature_temprb.append(day_name_backward_resInv)

                df_user_activity_train_temp_all[rate_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint16).values
                df_user_activity_train_temp_all[rate_name_backward_resInv] = (
                df_user_activity_train_temp_all[rate_name_backward] / \
                df_user_activity_train_temp_all[
                    "user_user_activity_register_time"]).astype(
                    np.float32)
                df_user_activity_train_temp_all[day_name_backward] = (df_gp_temp.apply(
                    lambda x: count_occurence(set(x), (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint8).values
                df_user_activity_train_temp_all[day_name_backward_resInv] = (
                df_user_activity_train_temp_all[day_name_backward] / \
                df_user_activity_train_temp_all[
                    "user_user_activity_register_time"]).astype(
                    np.float32)
                if rb != 0:
                    frequency_name_backward = "user_user_activity_frequency_b" + str(rb)
                    user_activity_feature_temprb.append(frequency_name_backward)
                    df_user_activity_train_temp_all[frequency_name_backward] = (
                        df_user_activity_train_temp_all[rate_name_backward] / (df_user_activity_train_temp_all[
                                                                                   day_name_backward] + 0.00000001)).astype(
                        np.float32)
                print("save user_activity_feature_temprb{}_file to {}".format(rb, user_activity_feature_temprb_file))
                df_user_activity_train_temp_all[user_activity_feature_temprb + ["user_id"]].to_csv(
                    user_activity_feature_temprb_file, header=True, index=False)
                df_user_activity_train_temp_all.drop(labels=user_activity_feature_temprb, axis=1, inplace=True)
                gc.collect()
                del df_gp_temp
                gc.collect()


        # df_user_activity_train.drop_duplicates(inplace=True)
        if not os.path.exists(temp_path + "user_activity_temp34.csv"):
            df_gp_temp = df_gp_temp_all["user_activity_day"]
            print("begin to get the user activity gap")
            df_user_activity_train_temp_all["user_user_activity_gap"] = (df_gp_temp.apply(
                lambda x: (len(set(x)) - 1) / (max(x) - min(x)) if len(set(x)) > 1 else min(x) / (
                    trainSpan[1] + min(x)))).astype(np.float32).values
            user_activity_gap_name_ls = ["user_user_activity_gap"]
            for g in tqdm(range(3, 13, 3)):
                gap_name = "user_user_activity_gap_b" + str(g)
                df_user_activity_train_temp_all[gap_name] = (df_gp_temp.apply(
                    lambda x: get_gap(x, trainSpan[1], g))).astype(np.float32).values
                user_activity_gap_name_ls.append(gap_name)
            user_activity_feature_temp34_file = temp_path + "user_activity_temp34.csv"
            user_activity_feature_temp34 = user_activity_gap_name_ls
            print("save user_activity_feature_temp34_file to  ", user_activity_feature_temp34_file)
            df_user_activity_train_temp_all[user_activity_feature_temp34 + ["user_id"]].to_csv(
                user_activity_feature_temp34_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp34, axis=1, inplace=True)
            del df_gp_temp
            gc.collect()
        if not os.path.exists(temp_path + "user_activity_temp35.csv"):
            print("begin to get the user activity day var")
            df_gp_temp = df_gp_temp_all["user_activity_day"]
            df_user_activity_train_temp_all["user_user_activity_day_var"] = (df_gp_temp.apply(
                lambda x: np.var(list(set(x))))).astype(np.float32).values
            user_activity_day_var_name_ls = ["user_user_activity_day_var"]
            for v in tqdm(range(3, 13, 3)):
                day_var_name = "user_user_activity_day_var_b" + str(v)
                df_user_activity_train_temp_all[day_var_name] = (df_gp_temp.apply(
                    lambda x: get_var(list(set(x)), trainSpan[1], v))).astype(np.float32).values
                user_activity_day_var_name_ls.append(day_var_name)
            user_activity_feature_temp35_file = temp_path + "user_activity_temp35.csv"
            user_activity_feature_temp35 = user_activity_day_var_name_ls
            print("save user_activity_feature_temp35_file to  ", user_activity_feature_temp35_file)
            df_user_activity_train_temp_all[user_activity_feature_temp35 + ["user_id"]].to_csv(
                user_activity_feature_temp35_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp35, axis=1, inplace=True)
            gc.collect()
            del df_gp_temp
            gc.collect()

        # df_gp_temp = df_gp_temp_all["user_user_activity_register_diff"]
        if not os.path.exists(temp_path + "user_activity_temp36.csv"):
            df_gp_temp = df_gp_temp_all["user_user_activity_register_diff"]
            print("get the timeseries feature of the user activity log after drop_dup(1)")
            df_user_activity_train_temp_all["user_user_activity_max"] = (df_gp_temp.apply(
                lambda x: max(x))).astype(
                np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_min"] = (df_gp_temp.apply(
                lambda x: min(x))).astype(
                np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_mean"] = (df_gp_temp.apply(
                lambda x: int(np.mean(list(x))))).astype(
                np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_median"] = (df_gp_temp.apply(
                lambda x: np.median(list(set(x))))).astype(
                np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_var_lt_std"] = (df_gp_temp.apply(
                lambda x: variance_larger_than_standard_deviation(list(set(x))))).astype(np.uint8).values

            user_activity_feature_temp36_file = temp_path + "user_activity_temp36.csv"
            user_activity_feature_temp36 = ["user_user_activity_max", "user_user_activity_min",
                                            "user_user_activity_mean", "user_user_activity_median",
                                            "user_user_activity_var_lt_std"]
            print("save user_activity_feature_temp36_file to  ", user_activity_feature_temp36_file)
            df_user_activity_train_temp_all[user_activity_feature_temp36 + ["user_id"]].to_csv(
                user_activity_feature_temp36_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp36, axis=1, inplace=True)
            del df_gp_temp
            gc.collect()
        if not os.path.exists(temp_path + "user_activity_temp37.csv"):
            print("get the timeseries feature of the user activity log after drop_dup(2)")
            df_gp_temp = df_gp_temp_all["user_user_activity_register_diff"]
            df_user_activity_train_temp_all["user_user_activity_tras"] = (df_gp_temp.apply(
                lambda x: time_reversal_asymmetry_statistic(x, 1))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_abs_energy"] = (df_gp_temp.apply(
                lambda x: abs_energy(x))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_cid_ce"] = (df_gp_temp.apply(
                lambda x: cid_ce(list(set(x))))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_mean_change"] = (df_gp_temp.apply(
                lambda x: mean_change(list(set(x))))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_kurtosis"] = (df_gp_temp.apply(
                lambda x: kurtosis(x))).astype(np.float32).values
            user_activity_feature_temp37_file = temp_path + "user_activity_temp37.csv"
            user_activity_feature_temp37 = [
                "user_user_activity_tras",
                "user_user_activity_abs_energy", "user_user_activity_cid_ce",
                "user_user_activity_mean_change", "user_user_activity_kurtosis"]
            print("save user_activity_feature_temp37_file to  ", user_activity_feature_temp37_file)
            df_user_activity_train_temp_all[user_activity_feature_temp37 + ["user_id"]].to_csv(
                user_activity_feature_temp37_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp37, axis=1, inplace=True)
            del df_gp_temp
            gc.collect()
        if not os.path.exists(temp_path + "user_activity_temp38.csv"):
            print("get the timeseries feature of the user activity log after drop_dup(3)")
            df_gp_temp = df_gp_temp_all["user_user_activity_register_diff"]
            df_user_activity_train_temp_all["user_user_activity_cam"] = (df_gp_temp.apply(
                lambda x: count_above_mean(list(set(x))))).astype(np.int8).values
            df_user_activity_train_temp_all["user_user_activity_cbm"] = (df_gp_temp.apply(
                lambda x: count_below_mean(list(set(x))))).astype(np.int8).values
            df_user_activity_train_temp_all["user_user_activity_bent"] = (df_gp_temp.apply(
                lambda x: binned_entropy(x, 4))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_autocorr"] = (df_gp_temp.apply(
                lambda x: autocorrelation(x, 1))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_rbrs"] = (df_gp_temp.apply(
                lambda x: ratio_beyond_r_sigma(x, 1))).astype(np.float32).values
            user_activity_feature_temp38_file = temp_path + "user_activity_temp38.csv"
            user_activity_feature_temp38 = [
                "user_user_activity_cam", "user_user_activity_cbm",
                "user_user_activity_bent", "user_user_activity_autocorr",
                "user_user_activity_rbrs"]
            print("save user_activity_feature_temp38_file to  ", user_activity_feature_temp38_file)
            df_user_activity_train_temp_all[user_activity_feature_temp38 + ["user_id"]].to_csv(
                user_activity_feature_temp38_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp38, axis=1,
                                                 inplace=True)
            del df_gp_temp
            gc.collect()

        if not os.path.exists(temp_path + "user_activity_temp39.csv"):
            df_gp_temp = df_gp_temp_all["user_activity_day"]
            df_user_activity_train_temp_all["user_user_activity_last_time"] = (df_gp_temp.apply(
                lambda x: trainSpan[1] - max(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_first_time"] = (df_gp_temp.apply(
                lambda x: trainSpan[1] - min(x))).astype(np.uint8).values
            user_activity_feature_temp39_file = temp_path + "user_activity_temp39.csv"
            user_activity_feature_temp39 = [
                "user_user_activity_register_time",
                "user_user_activity_last_time", "user_user_activity_first_time",
            ]
            print("save user_activity_feature_temp39_file to  ", user_activity_feature_temp39_file)
            df_user_activity_train_temp_all[user_activity_feature_temp39+["user_id"]].to_csv(
                user_activity_feature_temp39_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp39, axis=1,
                                                 inplace=True)
            del df_gp_temp
            gc.collect()
        if not os.path.exists(temp_path + "user_activity_temp40.csv"):
            df_gp_temp = df_gp_temp_all["user_activity_day"]
            df_user_activity_train_temp_all["user_user_activity_workday_rate"] = (df_gp_temp.apply(
                lambda x: get_workday_rate(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_workday_day"] = (df_gp_temp.apply(
                lambda x: get_workday_rate(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_workday_frequency"] = (df_user_activity_train_temp_all["user_user_activity_workday_rate"]/(df_user_activity_train_temp_all["user_user_activity_workday_day"]+0.00000001)).astype(np.float32)
            df_user_activity_train_temp_all["user_user_activity_weekend_rate"] = (df_gp_temp.apply(
                lambda x: get_weekend_rate(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_weekend_day"] = (df_gp_temp.apply(
                lambda x: get_weekend_day(x))).astype(np.uint8).values
            df_user_activity_train_temp_all["user_user_activity_weekend_frequency"] = (df_user_activity_train_temp_all["user_user_activity_weekend_rate"]/(df_user_activity_train_temp_all["user_user_activity_weekend_day"]+0.00000001)).astype(np.float32)
            user_activity_feature_temp40_file = temp_path + "user_activity_temp40.csv"
            user_activity_feature_temp40 = [
                "user_user_activity_workday_rate","user_user_activity_workday_day",
                "user_user_activity_workday_frequency", "user_user_activity_weekend_rate",
                "user_user_activity_weekend_day","user_user_activity_weekend_frequency"
            ]
            print("save user_activity_feature_temp40_file to  ", user_activity_feature_temp40_file)
            df_user_activity_train_temp_all[user_activity_feature_temp40+["user_id"]].to_csv(
                user_activity_feature_temp40_file, header=True, index=False)
            df_user_activity_train_temp_all.drop(labels=user_activity_feature_temp40, axis=1,
                                                 inplace=True)
            del df_gp_temp
            gc.collect()
        if not os.path.exists(temp_path + "user_activity_temp41.csv"):
            df_gp_temp = df_gp_temp_all["user_activity_day"]
            df_user_activity_train_temp_all["user_user_activity_workday_weekend_rate_ratio"] = (df_gp_temp.apply(
                lambda x: workday_weekend_rate_ratio(x))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_workday_weekend_day_ratio"] = (df_gp_temp.apply(
                lambda x: workday_weekend_day_ratio(x))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_only_in_weekend"] = (df_gp_temp.apply(
                lambda x: only_in_weekend(x))).astype(np.float32).values
            df_user_activity_train_temp_all["user_user_activity_mod_var"] = (df_gp_temp.apply(
                lambda x: get_mod_var(x))).astype(np.float32).values
            user_activity_feature_temp41_file = temp_path + "user_activity_temp41.csv"
            user_activity_feature_temp41 = [
                "user_user_activity_workday_weekend_rate_ratio",
                "user_user_activity_workday_weekend_day_ratio",
                "user_user_activity_only_in_weekend", "user_user_activity_mod_var",
            ]
            print("save user_activity_feature_temp41_file to  ", user_activity_feature_temp41_file)
            df_user_activity_train_temp_all[user_activity_feature_temp41+["user_id"]].to_csv(
                user_activity_feature_temp41_file, header=True, index=False)
            print(df_user_activity_train_temp_all.describe())
            del df_gp_temp, df_gp_temp_all, df_user_activity_train
            gc.collect()
        else:
            df_user_activity_train_temp_all = pd.read_csv(temp_path + "user_activity_temp41.csv", header=0,
                                                          index_col=None)
            print(df_user_activity_train_temp_all.describe())
            del df_gp_temp_all, df_user_activity_train
            gc.collect()
        for i in range(1, 41):
            user_activity_feature_temp_file = temp_path + "user_activity_temp" + str(i) + ".csv"
            df_user_activity_train_temp = pd.read_csv(user_activity_feature_temp_file, header=0, index_col=None)
            print(df_user_activity_train_temp.describe())
            df_user_activity_train_temp_all = df_user_activity_train_temp_all.merge(df_user_activity_train_temp,
                                                                                    how="left")
            del df_user_activity_train_temp
            gc.collect()
        # df_user_activity_train = df_user_activity_train[user_activity_feature]
        print("begin to get interaction feature within activity log")
        df_user_activity_train_temp_all["video_author_ratio"] = (
        df_user_activity_train_temp_all["user_video_rate"] / df_user_activity_train_temp_all[
            "user_author_rate"]).astype(np.float32)
        df_user_activity_train_temp_all["user_video_ratio"] = (
        df_user_activity_train_temp_all["user_video_rate"] / df_user_activity_train_temp_all[
            "user_user_activity_day_f0"]).astype(np.float32)
        df_user_activity_train_temp_all["user_author_ratio"] = (
        df_user_activity_train_temp_all["user_author_rate"] / df_user_activity_train_temp_all[
            "user_user_activity_day_f0"]).astype(np.float32)
        print("begin to drop the low variance features for the user_activity feature")
        lsc = get_low_std_cols(df_user_activity_train_temp_all)
        df_user_activity_train_temp_all.drop(labels=lsc, axis=1, inplace=True)
        print("begin to drop the columns with missing value ration gt 50% in user_activity log")
        mrc = get_missing_ratio_cols(df_user_activity_train_temp_all)
        df_user_activity_train_temp_all.drop(labels=mrc, axis=1, inplace=True)
        gc.collect()
        # ds4 = df_user_activity_train_temp_all.describe()
        print(df_user_activity_train_temp_all.describe())
        print("memory usage", df_user_activity_train_temp_all.memory_usage().sum() / 1024 ** 2)
        print(
            "finish getting basic user activity log features, get {} features ".format(
                len(df_user_activity_train_temp_all.columns) - 1))
        file_name4 = "../input/fusai/basic_user_activity_feature_" + str(trainSpan[0]) + "_" + str(
            trainSpan[1]) + ".csv"
        print("save the basic feature to ", file_name4)
        df_user_activity_train_temp_all.to_csv(file_name4, header=True, index=False)
        return df_user_activity_train_temp_all

def preprocess():
    gc.enable()
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.int8, "register_type": np.int8,
                           "device_type": np.uint16}
    df_user_register = pd.read_table("/mnt/datasets/fusai/user_register_log.txt", header=None, names=user_register_log,
                                     index_col=None, dtype=dtype_user_register)

    video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.int8}
    df_video_create = pd.read_table("/mnt/datasets/fusai/video_create_log.txt", header=None, names=video_create_log,
                                    index_col=None, dtype=dtype_video_create)

    app_launch_log = ["user_id", "app_launch_day"]
    dtype_app_launch = {"user_id": np.uint32, "app_launch_day": np.int8}
    df_app_launch = pd.read_table("/mnt/datasets/fusai/app_launch_log.txt", header=None, names=app_launch_log,
                                  index_col=None, dtype=dtype_app_launch)
    print("df_user_register    :", df_user_register.shape)
    print("df_video_create     :", df_video_create.shape)
    print("df_app_launch        :", df_app_launch.shape)
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.float16, "page": np.float16,
                           "video_id": object,
                           "author_id": object, "action_type": np.float16}
    df_user_activity_id = pd.read_table("/mnt/datasets/fusai/user_activity_log.txt", header=None,
                                        names=user_activity_log, index_col=None, dtype=dtype_user_activity)
    print("df_user_activity    :", df_user_activity_id.shape)

    df_user_activity_id.dropna(inplace=True, axis=0)
    df_user_activity_id["video_id"] = pd.to_numeric(df_user_activity_id["video_id"], downcast='integer').astype(
        np.uint32)
    df_user_activity_id["author_id"] = pd.to_numeric(df_user_activity_id["author_id"], downcast='integer').astype(
        np.uint32)
    df_user_activity_id["user_activity_day"] = pd.to_numeric(df_user_activity_id["user_activity_day"],
                                                             downcast='integer').astype(np.int8)
    df_user_activity_id["page"] = pd.to_numeric(df_user_activity_id["page"], downcast='integer').astype(np.int8)
    df_user_activity_id["action_type"] = pd.to_numeric(df_user_activity_id["action_type"], downcast='integer').astype(
        np.int8)
    gc.collect()

    user_outliers = df_user_register[(df_user_register["device_type"] == 1) | (
    (df_user_register["register_day"].isin([24, 25, 26])) & (df_user_register["register_type"] == 3) & (
    (df_user_register["device_type"] == 10566) | (df_user_register["device_type"] == 3036)))][
        "user_id"].unique().tolist()
    print(len(user_outliers))
    gc.collect()
    df_user_register = df_user_register[~df_user_register["user_id"].isin(user_outliers)]
    df_app_launch = df_app_launch[~df_app_launch["user_id"].isin(user_outliers)]
    df_video_create = df_video_create[~df_video_create["user_id"].isin(user_outliers)]
    gc.collect()
    df_user_activity_id = df_user_activity_id[~df_user_activity_id["user_id"].isin(user_outliers)]
    gc.collect()
    print("df_user_register    :", df_user_register.shape)
    print("df_video_create     :", df_video_create.shape)
    print("df_app_launch        :", df_app_launch.shape)
    print("df_user_activity    :", df_user_activity_id.shape)
    df_user_activity_id.to_csv("../input/fusai/user_activity_log.txt", index=False, header=True)
    df_user_register.to_csv("../input/fusai/user_register_log.txt", index=False, header=True)
    df_app_launch.to_csv("../input/fusai/app_launch_log.txt", index=False, header=True)
    df_video_create.to_csv("../input/fusai/video_create_log.txt", index=False, header=True)
    user_outliers = df_user_register[(df_user_register["device_type"] == 1) | (
    (df_user_register["register_day"].isin([24, 25, 26])) & (df_user_register["register_type"] == 3) & (
    (df_user_register["device_type"] == 10566) | (df_user_register["device_type"] == 3036)))][
        "user_id"].unique().tolist()
    print(len(user_outliers))
    df_outlier = pd.DataFrame()
    df_outlier["user_id"] = pd.Series(user_outliers)
    df_outlier["proba"] = 0
    df_outlier.to_csv("../work/outliers.csv", header=True, index=False)
    del df_user_activity_id, df_user_register, df_app_launch, df_video_create
    gc.collect()

def processing(trainSpan=(8, 23), label=True):
    if label:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 23 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 23
    else:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 30 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 30

    if not os.path.exists("../input/fusai/user_register_log.txt"):
        preprocess()
    print("get users from user register log")

    dtype_user_register = {"user_id": np.uint32, "register_day": np.int8, "register_type": np.int8,
                           "device_type": np.uint16}
    user_len = pd.read_csv("../input/fusai/user_register_log.txt", index_col=None,
                           dtype=dtype_user_register, usecols=[0]).iloc[:, 0].nunique()
    if user_len > 560000:
        preprocess()
    df_user_register = pd.read_csv("../input/fusai/user_register_log.txt", header=0, index_col=None,
                                   dtype=dtype_user_register)
    # df_user_register_base = df_user_register[["user_id", "register_day"]].drop_duplicates()
    # del df_user_register
    # gc.collect()
    df_ls = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=16) as Executor:
        # process_pool = Executor(max_workers=5)
        tfps = []
        for wh in ["reg", "app", "video", "act"]:
            tfp = Executor.submit(create_feature, which=wh, trainSpan=trainSpan, base=df_user_register)
            tfps.append(tfp)
        for tfp in concurrent.futures.as_completed(tfps):
            try:
                res = tfp.result()
                df_ls.append(res)
                print(res.describe())
            except Exception as e:
                print('process: feature creating process error. ' + str(e))
            else:
                print('process: feature engineering process ok.')
    print('feature creating finished! begin to merge the individual data.')
    del df_user_register
    gc.collect()
    assert len(df_ls) == 4, "the output of the multi-process must be a list of length 4"
    for df in df_ls:
        if "register" in list(df.columns)[3]:
            df_user_register_train = df
            # print(df_user_register_train.describe())
        elif "launch" in list(df.columns)[3]:
            df_app_launch_train = df
            # print(df_app_launch_train.describe())
        elif "video" in list(df.columns)[3]:
            df_video_create_train = df
            # print(df_video_create_train.describe())
        elif "activity" in list(df.columns)[3]:
            df_user_activity_train = df
            # print(df_user_activity_train.describe())
    if label:
        print("get users from app launch log")
        # app_launch_log = ["user_id","app_launch_day"]
        dtype_app_launch = {
            "user_id": np.uint32,
            "app_launch_day": np.int8,
        }
        df_app_launch = pd.read_csv("../input/fusai/app_launch_log.txt", header=0, index_col=None,
                                    dtype=dtype_app_launch)
        # active_user_register = (df_user_register.loc[(df_user_register["register_day"]>trainSpan[1])&(df_user_register["register_day"]<=(trainSpan[1]+7))]).user_id.unique().tolist()
        active_app_launch = (df_app_launch.loc[(df_app_launch["app_launch_day"] > trainSpan[1]) & (df_app_launch["app_launch_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        print("get users from video create")
        dtype_video_create = {"user_id": np.uint32, "video_create_day": np.int8}
        df_video_create = pd.read_csv("../input/fusai/video_create_log.txt", header=0, index_col=None,
                                      dtype=dtype_video_create)
        active_video_create = (df_video_create.loc[(df_video_create["video_create_day"]>trainSpan[1])&(df_video_create["video_create_day"]<=(trainSpan[1]+7))]).user_id.unique().tolist()
        print("get users from user activity log")
        dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.int8, "page": np.int8,
                               "video_id": np.uint32,
                               "author_id": np.uint32, "action_type": np.int8}
        df_user_activity = pd.read_csv("../input/fusai/user_activity_log.txt", header=0, index_col=None,
                                       dtype=dtype_user_activity,usecols=["user_id","user_activity_day"])
        active_user_activity = (df_user_activity.loc[(df_user_activity["user_activity_day"] > trainSpan[1]) & (df_user_activity["user_activity_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user = list(set(active_app_launch+active_video_create+active_user_activity))
        del df_app_launch,df_video_create,df_user_activity
        gc.collect()

        df_user_register_train["label"] = 0
        df_user_register_train.loc[df_user_register_train["user_id"].isin(active_user), "label"] = 1

        # df_app_launch_train["label"] = 0
        # df_app_launch_train.loc[df_app_launch_train["user_id"].isin(active_user), "label"] = 1
        #
        # df_video_create_train["label"] = 0
        # df_video_create_train.loc[df_video_create_train["user_id"].isin(active_user), "label"] = 1
        #
        # df_user_activity_train["label"] = 0
        # df_user_activity_train.loc[df_user_activity_train["user_id"].isin(active_user), "label"] = 1
    df_user_register_train = df_user_register_train.loc[
        (df_user_register_train["register_day"] >= 0) & (df_user_register_train["register_day"] <= trainSpan[1])]
    # df_launch_register = df_app_launch_train.merge(df_user_register_train, how="left").fillna(0)
    # df_launch_register = df_user_register_train.merge(df_app_launch_train, how="left").fillna(0)
    df_launch_register = df_user_register_train.merge(df_app_launch_train, how="left")
    # print(df_register_launch.describe())
    # df_launch_register_create = df_launch_register.merge(df_video_create_train, how="left").fillna(0)
    df_launch_register_create = df_launch_register.merge(df_video_create_train, how="left")
    # print(df_register_launch_create.describe())
    df_launch_activity_register_create = df_launch_register_create.merge(df_user_activity_train, how="left").fillna(0)
    print("begin to creat interactive features between logs")
    df_launch_activity_register_create["user_app_launch_video_ratio"] = (
    df_launch_activity_register_create["user_video_create_day_f0"] / (
    df_launch_activity_register_create["user_app_launch_rate_f0"] + 0.00000001)).astype(np.float32)
    df_launch_activity_register_create["user_app_launch_activity_ratio"] = (
    df_launch_activity_register_create["user_user_activity_day_f0"] / (
    df_launch_activity_register_create["user_app_launch_rate_f0"] + 0.00000001)).astype(np.float32)
    df_launch_activity_register_create["register_day"] = df_launch_activity_register_create["register_day"] % 7
    # df_launch_activity_register_create["user_activity_video_ratio"] = df_launch_activity_register_create["user_video_create_day"]/df_launch_activity_register_create["user_user_activity_day"]

    print("the description of the final feature array ")
    print(df_launch_activity_register_create.describe())
    keep_feature = df_launch_activity_register_create.columns
    print("final feature length is {}".format(len(keep_feature)))
    print("memory usage", df_launch_activity_register_create.memory_usage().sum() / 1024 ** 2)
    file_name5 = "../input/final2_feature_" + str(trainSpan[0]) + "_" + str(trainSpan[1]) + ".csv"
    df_launch_activity_register_create.to_csv(file_name5, header=True, index=False)
    del df_user_register_train,  df_user_activity_train, df_launch_register, df_app_launch_train, df_launch_register_create, df_video_create_train, df_ls
    gc.collect()
    # ds5.to_csv("kuaishou_stats2.csv", mode='a')
    return df_launch_activity_register_create

# if __name__=="__main__":
#     train_set = processing((1,15),label=True)
#     print(train_set.info())
