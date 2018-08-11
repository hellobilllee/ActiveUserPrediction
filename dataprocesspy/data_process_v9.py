from collections import Counter
import gc
import pandas as pd
import numpy as np

user_register_log = ["user_id", "register_day", "register_type", "device_type"]
app_launch_log = ["user_id", "app_launch_day"]
video_create_log = ["user_id", "video_create_day"]
user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]


def count_occurence(x, span):
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


def get_var(ls, maxSpan, v):
    la = [i for i in ls if i > (maxSpan - v)]
    if la:
        return np.var(la)
    else:
        return 0


def get_ratio(ls, e):
    return ls.count(e) * 1.0 / len(ls)


def processing(trainSpan=(1, 23), label=True):
    if label:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 23 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 23
    else:
        assert isinstance(trainSpan, tuple), "input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0] > 0 and trainSpan[0] < 30 and trainSpan[1] > trainSpan[0] and trainSpan[1] <= 30

    print("get users from user register log")
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8,
                           "device_type": np.uint32}
    df_user_register = pd.read_csv("data/A2/user_register_log.txt", header=0, index_col=None,
                                   dtype=dtype_user_register)
    user_outliers = df_user_register[(df_user_register["register_day"]==24)&(df_user_register["register_type"]==3)&((df_user_register["device_type"]==1)|(df_user_register["device_type"]==223)|(df_user_register["device_type"]==83))]["user_id"].unique().tolist()
    df_user_register = df_user_register[~df_user_register["user_id"].isin(user_outliers)]
    df_user_register_train = df_user_register.loc[
        (df_user_register["register_day"] >= trainSpan[0]) & (df_user_register["register_day"] <= trainSpan[1])]

    df_user_register_train["register_day_rate"] = (
    df_user_register_train.groupby(by=["register_day"])["register_day"].transform("count")).astype(np.uint16)
    df_user_register_train["register_day_type_rate"] = (
    df_user_register_train.groupby(by=["register_day", "register_type"])["register_type"].transform("count")).astype(
        np.uint16)
    df_user_register_train["register_day_type_ratio"] = (
    df_user_register_train["register_day_type_rate"] / df_user_register_train["register_day_rate"]).astype(np.float32)
    df_user_register_train["register_day_device_rate"] = (
    df_user_register_train.groupby(by=["register_day", "device_type"])["device_type"].transform("count")).astype(
        np.uint16)
    df_user_register_train["register_day_device_ratio"] = (
    df_user_register_train["register_day_device_rate"] / df_user_register_train["register_day_rate"]).astype(np.float32)
    df_user_register_train["register_type_rate"] = (
    df_user_register_train.groupby(by=["register_type"])["register_type"].transform("count")).astype(np.uint16)
    df_user_register_train["register_type_ratio"] = (
    df_user_register_train["register_type_rate"] / len(df_user_register_train)).astype(np.float32)
    df_user_register_train["register_type_device"] = (
    df_user_register_train.groupby(by=["register_type"])["device_type"].transform(lambda x: x.nunique())).astype(
        np.uint16)
    df_user_register_train["register_type_device_rate"] = (
    df_user_register_train.groupby(by=["register_type", "device_type"])["device_type"].transform("count")).astype(
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
    df_user_register_train.groupby(by=["device_type", "register_type"])["register_type"].transform("count")).astype(
        np.uint16)
    df_user_register_train["device_type_register_ratio"] = (
    df_user_register_train["device_type_register_rate"] / df_user_register_train["device_type_rate"]).astype(np.float32)
    df_user_register_train["register_day_register_type_device_rate"] = (
    df_user_register_train.groupby(by=["register_day", "register_type", "device_type"])["device_type"].transform(
        "count")).astype(np.uint16)
    df_user_register_train["register_day_register_type_device_ratio"] = (
    df_user_register_train["register_day_register_type_device_rate"] / df_user_register_train[
        "register_day_type_rate"]).astype(np.float32)
    df_user_register_train["register_day_device_type_register_rate"] = (
    df_user_register_train.groupby(by=["register_day", "device_type", "register_type"])["register_type"].transform(
        "count")).astype(np.uint16)
    df_user_register_train["register_day_device_type_register_ratio"] = (
    df_user_register_train["register_day_device_type_register_rate"] / df_user_register_train[
        "register_day_device_rate"]).astype(np.float32)

    user_register_feature = ["user_id",
                             "register_day_type_rate",
                             "register_day_type_ratio",
                             "register_day_device_ratio",
                             "register_type_ratio",
                             "register_type_device",
                             "register_type_device_ratio",
                             "register_day_device_rate",
                             "device_type_ratio",
                             "device_type_register_ratio",
                             "register_day_register_type_device_ratio",
                             "register_day_device_type_register_ratio"
                             ]
    df_user_register_base = df_user_register[["user_id", "register_day"]].drop_duplicates()
    df_user_register_train = df_user_register_train[user_register_feature].drop_duplicates()
    ds1 = df_user_register_train.describe()
    print(ds1)
    ds1.to_csv("kuaishou_stats2.csv", mode='a')
    # df_user_register_train.to_csv("data/user_register_temp.csv",header=True,index=False)
    # del df_user_register_train
    # gc.collect()

    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {
        "user_id": np.uint32,
        "app_launch_day": np.uint8,
    }
    df_app_launch = pd.read_csv("data/A2/app_launch_log.txt", header=0, index_col=None, dtype=dtype_app_launch)
    df_app_launch = df_app_launch[~df_app_launch["user_id"].isin(user_outliers)]
    df_app_launch = df_app_launch.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    df_app_launch_train = df_app_launch.loc[
        (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]
    # print(df_app_launch_train.describe())
    df_app_launch_train["user_app_launch_register_time"] = (df_app_launch_train["register_day"].apply(
        lambda x: (trainSpan[1] - x + 1))).astype(np.uint8)

    df_app_launch_train["user_app_launch_rate"] = (df_app_launch_train.groupby(by=["user_id"])[
                                                       "app_launch_day"].transform("count")).astype(np.uint8)
    df_app_launch_train["user_app_launch_rate_spanInv"] = (
    df_app_launch_train["user_app_launch_rate"] * 1.0 / (trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_app_launch_train["user_app_launch_rate_resInv"] = (
    df_app_launch_train["user_app_launch_rate"] * 1.0 / df_app_launch_train["user_app_launch_register_time"]).astype(
        np.float32)
    df_app_launch_train["user_app_launch_rate_spanResInv"] = (
    df_app_launch_train["user_app_launch_rate_spanInv"] * 1.0 / df_app_launch_train[
        "user_app_launch_register_time"]).astype(np.float32)
    df_app_launch_train["user_app_launch_last_time"] = (
    df_app_launch_train.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: trainSpan[1] - max(x))).astype(
        np.uint8)
    app_launch_rate_name_forward_ls = []
    for rf in range(1, 7, 1):
        rate_name_forward = "user_app_launch_rate_f" + str(rf)
        app_launch_rate_name_forward_ls.append(rate_name_forward)
        df_app_launch_train[rate_name_forward] = (df_app_launch_train.groupby(by=["user_id"])[
                                                      "app_launch_day"].transform(
            lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf)))).astype(np.uint8)
        rate_name_forward_spanInv = rate_name_forward + "_spanInv"
        app_launch_rate_name_forward_ls.append(rate_name_forward_spanInv)
        df_app_launch_train[rate_name_forward_spanInv] = (
        df_app_launch_train[rate_name_forward] * 1.0 / (trainSpan[1] - trainSpan[0] - rf)).astype(np.float32)
        rate_name_forward_resInv = rate_name_forward + "_resInv"
        app_launch_rate_name_forward_ls.append(rate_name_forward_resInv)
        df_app_launch_train[rate_name_forward_resInv] = (
        df_app_launch_train[rate_name_forward] * 1.0 / df_app_launch_train["user_app_launch_register_time"]).astype(
            np.float32)
        rate_name_forward_spanResInv = rate_name_forward + "_spanResInv"
        app_launch_rate_name_forward_ls.append(rate_name_forward_spanResInv)
        df_app_launch_train[rate_name_forward_spanResInv] = (
        df_app_launch_train[rate_name_forward_spanInv] * 1.0 / df_app_launch_train[
            "user_app_launch_register_time"]).astype(np.float32)
    app_launch_rate_name_backward_ls = []
    for rb in range(14, -1, -1):
        rate_name_backward = "user_app_launch_rate_b" + str(rb)
        app_launch_rate_name_backward_ls.append(rate_name_backward)
        df_app_launch_train[rate_name_backward] = (df_app_launch_train.groupby(by=["user_id"])[
                                                       "app_launch_day"].transform(
            lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint8)
        rate_name_backward_resInv = rate_name_backward + "_resInv"
        app_launch_rate_name_backward_ls.append(rate_name_backward_resInv)
        df_app_launch_train[rate_name_backward_resInv] = (
        df_app_launch_train[rate_name_backward] * 1.0 / df_app_launch_train["user_app_launch_register_time"]).astype(
            np.float32)

    df_app_launch_train["user_app_launch_gap"] = (df_app_launch_train.groupby(by=["user_id"])[
                                                      "app_launch_day"].transform(
        lambda x: (len(set(x)) - 1) * 1.0 / (max(x) - min(x)) if len(set(x)) > 1 else min(x) * 1.0 / (
        trainSpan[1] + min(x)))).astype(np.float32)
    app_launch_gap_name_ls = []
    for g in range(5, 12, 3):
        gap_name = "user_app_launch_gap_b" + str(g)
        df_app_launch_train[gap_name] = (df_app_launch_train.groupby(by=["user_id"])[
                                             "app_launch_day"].transform(lambda x: get_gap(x, trainSpan[1], g))).astype(
            np.float32)
        app_launch_gap_name_ls.append(gap_name)

    df_app_launch_train["user_app_launch_var"] = (df_app_launch_train.groupby(by=["user_id"])[
                                                      "app_launch_day"].transform(
        lambda x: np.var(list(set(x))))).astype(np.float32)
    app_launch_var_name_ls = []

    for v in range(5, 12, 3):
        var_name = "user_app_launch_var_b" + str(v)
        df_app_launch_train[var_name] = (df_app_launch_train.groupby(by=["user_id"])[
                                             "app_launch_day"].transform(
            lambda x: get_var(list(set(x)), trainSpan[1], v))).astype(np.float32)
        app_launch_var_name_ls.append(var_name)

    app_launch_feature = [
                             "user_id",
                             "user_app_launch_rate",
                             "user_app_launch_rate_spanInv",
                             "user_app_launch_rate_resInv",
                             "user_app_launch_rate_spanResInv",
                             "user_app_launch_gap",
                             "user_app_launch_var",
                             "user_app_launch_last_time",
                         ] \
                         + app_launch_rate_name_forward_ls \
                         + app_launch_rate_name_backward_ls \
                         + app_launch_gap_name_ls \
                         + app_launch_var_name_ls
    # print(app_launch_feature)
    df_app_launch_train = df_app_launch_train[app_launch_feature].drop_duplicates()
    ds2 = df_app_launch_train.describe()
    print(ds2)
    ds2.to_csv("kuaishou_stats2.csv", mode='a')
    # df_app_launch_train.to_csv("data/user_app_launch_temp.csv",header=True,index=False)
    # del df_app_launch_train
    # gc.collect()

    print("get users from video create")
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_csv("data/A2/video_create_log.txt", header=0, index_col=None,
                                  dtype=dtype_video_create)
    df_video_create = df_video_create[~df_video_create["user_id"].isin(user_outliers)]
    df_video_create = df_video_create.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)

    df_video_create_train = df_video_create.loc[
        (df_video_create["video_create_day"] >= trainSpan[0]) & (df_video_create["video_create_day"] <= trainSpan[1])]
    df_video_create_train["user_video_create_register_time"] = (
    df_video_create_train["register_day"].apply(lambda x: (trainSpan[1] - x + 1))).astype(np.uint8)

    df_video_create_train["user_video_create_rate"] = (df_video_create_train.groupby(by=["user_id"])[
                                                           "video_create_day"].transform("count")).astype(np.uint16)
    df_video_create_train["user_video_create_rate_spanInv"] = (
    df_video_create_train["user_video_create_rate"] * 1.0 / (trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_video_create_train["user_video_create_rate_resInv"] = (
    df_video_create_train["user_video_create_rate"] * 1.0 / df_video_create_train[
        "user_video_create_register_time"]).astype(np.float32)
    df_video_create_train["user_video_create_rate_spanResInv"] = (
    df_video_create_train["user_video_create_rate_spanInv"] * 1.0 / df_video_create_train[
        "user_video_create_register_time"]).astype(np.float32)
    df_video_create_train["user_video_create_day"] = (df_video_create_train.groupby(by=["user_id"])[
                                                          "video_create_day"].transform(lambda x: x.nunique())).astype(
        np.uint8)
    df_video_create_train["user_video_create_day_spanInv"] = (
    df_video_create_train["user_video_create_day"] * 1.0 / (trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_video_create_train["user_video_create_day_resInv"] = (
    df_video_create_train["user_video_create_day"] * 1.0 / df_video_create_train[
        "user_video_create_register_time"]).astype(np.float32)
    df_video_create_train["user_video_create_day_spanResInv"] = (
    df_video_create_train["user_video_create_day_spanInv"] * 1.0 / df_video_create_train[
        "user_video_create_register_time"]).astype(np.float32)
    df_video_create_train["user_video_create_frequency"] = (df_video_create_train["user_video_create_rate"] * 1.0 / \
                                                            df_video_create_train["user_video_create_day"]).astype(
        np.float32)
    df_video_create_train["user_video_create_last_time"] = (
    df_video_create_train.groupby(by=["user_id"])["video_create_day"].transform(
        lambda x: trainSpan[1] - max(x))).astype(np.uint8)
    video_create_rate_name_forward_ls = []
    for rf in range(1, 7, 2):
        rate_name_forward = "user_video_create_rate_f" + str(rf)
        day_name_forward = "user_video_create_day_f" + str(rf)
        frequency_name_forward = "user_video_create_frequency_f" + str(rf)
        video_create_rate_name_forward_ls.append(rate_name_forward)
        video_create_rate_name_forward_ls.append(day_name_forward)
        video_create_rate_name_forward_ls.append(frequency_name_forward)
        df_video_create_train[rate_name_forward] = (df_video_create_train.groupby(by=["user_id"])[
            "video_create_day"].transform(
            lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf)))).astype(np.uint16)
        df_video_create_train[day_name_forward] = (df_video_create_train.groupby(by=["user_id"])[
            "video_create_day"].transform(
            lambda x: count_occurence(set(x), (trainSpan[0], trainSpan[1] - rf)))).astype(np.uint8)
        df_video_create_train[frequency_name_forward] = (
        df_video_create_train[rate_name_forward] * 1.0 / df_video_create_train[day_name_forward]).astype(np.float32)
        rate_name_forward_spanInv = rate_name_forward + "_spanInv"
        day_name_forward_spanInv = day_name_forward + "_spanInv"
        video_create_rate_name_forward_ls.append(rate_name_forward_spanInv)
        video_create_rate_name_forward_ls.append(day_name_forward_spanInv)
        df_video_create_train[rate_name_forward_spanInv] = (df_video_create_train[rate_name_forward] * 1.0 / (
            trainSpan[1] - trainSpan[0] - rf)).astype(np.float32)
        df_video_create_train[day_name_forward_spanInv] = (df_video_create_train[day_name_forward] * 1.0 / (
            trainSpan[1] - trainSpan[0] - rf)).astype(np.float32)
        rate_name_forward_resInv = rate_name_forward + "_resInv"
        day_name_forward_resInv = day_name_forward + "_resInv"
        video_create_rate_name_forward_ls.append(rate_name_forward_resInv)
        video_create_rate_name_forward_ls.append(day_name_forward_resInv)
        df_video_create_train[rate_name_forward_resInv] = (
        df_video_create_train[rate_name_forward] * 1.0 / df_video_create_train[
            "user_video_create_register_time"]).astype(np.float32)
        df_video_create_train[day_name_forward_resInv] = (
        df_video_create_train[day_name_forward] * 1.0 / df_video_create_train[
            "user_video_create_register_time"]).astype(np.float32)
        rate_name_forward_spanResInv = rate_name_forward + "_spanResInv"
        day_name_forward_spanResInv = day_name_forward + "_spanResInv"
        video_create_rate_name_forward_ls.append(rate_name_forward_spanResInv)
        video_create_rate_name_forward_ls.append(day_name_forward_spanResInv)
        df_video_create_train[rate_name_forward_spanResInv] = (df_video_create_train[rate_name_forward_spanInv] * 1.0 / \
                                                               df_video_create_train[
                                                                   "user_video_create_register_time"]).astype(
            np.float32)
        df_video_create_train[day_name_forward_spanResInv] = (df_video_create_train[day_name_forward_spanInv] * 1.0 / \
                                                              df_video_create_train[
                                                                  "user_video_create_register_time"]).astype(np.float32)
    video_create_rate_name_backward_ls = []
    for rb in range(12, -1, -3):
        rate_name_backward = "user_video_create_rate_b" + str(rb)
        video_create_rate_name_backward_ls.append(rate_name_backward)
        day_name_backward = "user_video_create_day_b" + str(rb)
        video_create_rate_name_backward_ls.append(day_name_backward)
        if rb != 0:
            frequency_name_backward = "user_video_create_frequency_b" + str(rb)
            video_create_rate_name_backward_ls.append(frequency_name_backward)
        df_video_create_train[rate_name_backward] = (df_video_create_train.groupby(by=["user_id"])[
            "video_create_day"].transform(
            lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint16)
        df_video_create_train[day_name_backward] = (df_video_create_train.groupby(by=["user_id"])[
            "video_create_day"].transform(
            lambda x: count_occurence(set(x), (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint8)
        if rb != 0:
            df_video_create_train[frequency_name_backward] = (
            df_video_create_train[rate_name_backward] * 1.0 / df_video_create_train[day_name_backward]).astype(
                np.float32)
        rate_name_backward_resInv = rate_name_backward + "_resInv"
        day_name_backward_resInv = day_name_backward + "_resInv"
        video_create_rate_name_backward_ls.append(rate_name_backward_resInv)
        video_create_rate_name_backward_ls.append(day_name_backward_resInv)
        df_video_create_train[rate_name_backward_resInv] = (
        df_video_create_train[rate_name_backward] * 1.0 / df_video_create_train[
            "user_video_create_register_time"]).astype(np.float32)
        df_video_create_train[day_name_backward_resInv] = (
        df_video_create_train[day_name_backward] * 1.0 / df_video_create_train[
            "user_video_create_register_time"]).astype(np.float32)

    df_video_create_train["user_video_create_gap"] = (df_video_create_train.groupby(by=["user_id"])[
                                                          "video_create_day"].transform(
        lambda x: (len(set(x)) - 1) * 1.0 / (max(x) - min(x)) if len(set(x)) > 1 else min(x) * 1.0 / (
        trainSpan[1] + min(x))) * 1.0).astype(np.float32)
    video_create_gap_name_ls = []
    for g in range(5, 12, 3):
        gap_name = "user_video_create_gap_b" + str(g)
        df_video_create_train[gap_name] = (df_video_create_train.groupby(by=["user_id"])[
                                               "video_create_day"].transform(
            lambda x: get_gap(x, trainSpan[1], g)) * 1.0).astype(np.float32)
        video_create_gap_name_ls.append(gap_name)

    df_video_create_train["user_video_create_day_var"] = (df_video_create_train.groupby(by=["user_id"])[
                                                              "video_create_day"].transform(
        lambda x: np.var(list(set(x)))) * 1.0).astype(np.float32)
    df_video_create_train["user_video_create_var"] = (df_video_create_train.groupby(by=["user_id"])[
                                                          "video_create_day"].transform(
        lambda x: np.var(list(x))) * 1.0).astype(np.float32)
    video_create_var_name_ls = []
    for v in range(5, 12, 3):
        var_name = "user_video_create_var_b" + str(v)
        day_var_name = "user_video_create_day_var_b" + str(v)
        df_video_create_train[day_var_name] = (df_video_create_train.groupby(by=["user_id"])[
                                                   "video_create_day"].transform(
            lambda x: get_var(list(set(x)), trainSpan[1], v)) * 1.0).astype(np.float32)
        df_video_create_train[var_name] = (df_video_create_train.groupby(by=["user_id"])[
                                               "video_create_day"].transform(
            lambda x: get_var(list(x), trainSpan[1], v)) * 1.0).astype(np.float32)
        video_create_var_name_ls.append(day_var_name)
        video_create_var_name_ls.append(var_name)

    # print(df_video_create_train.describe())
    video_create_feature = ["user_id",
                            "user_video_create_rate",
                            "user_video_create_rate_spanInv",
                            "user_video_create_rate_resInv",
                            "user_video_create_rate_spanResInv",
                            "user_video_create_day",
                            "user_video_create_day_spanInv",
                            "user_video_create_day_resInv",
                            "user_video_create_day_spanResInv",
                            "user_video_create_frequency",
                            "user_video_create_gap",
                            "user_video_create_day_var",
                            "user_video_create_var",
                            "user_video_create_last_time",
                            ] \
                           + video_create_rate_name_forward_ls \
                           + video_create_rate_name_backward_ls \
                           + video_create_gap_name_ls \
                           + video_create_var_name_ls

    df_video_create_train = df_video_create_train[video_create_feature].drop_duplicates()
    ds3 = df_video_create_train.describe()
    print(ds3)
    ds3.to_csv("kuaishou_stats2.csv", mode='a')
    # df_video_create_train.to_csv("data/user_video_create_temp.csv",header=True,index=False)
    # del df_video_create_train
    # gc.collect()

    print("get users from user activity log")
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id": np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_csv("data/A2/user_activity_log.txt", header=0, index_col=None,
                                   dtype=dtype_user_activity)
    df_user_activity = df_user_activity[~df_user_activity["user_id"].isin(user_outliers)]
    df_user_activity = df_user_activity.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)

    df_user_activity_train = df_user_activity.loc[
        (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
            df_user_activity["user_activity_day"] <= trainSpan[1])]

    active_user = []
    if label:
        # active_user_register = (df_user_register.loc[(df_user_register["register_day"] > trainSpan[1]) & (
        # df_user_register["register_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_app_launch = (df_app_launch.loc[(df_app_launch["app_launch_day"] > trainSpan[1]) & (
        df_app_launch["app_launch_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_video_create = (df_video_create.loc[(df_video_create["video_create_day"] > trainSpan[1]) & (
        df_video_create["video_create_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user_activity = (df_user_activity.loc[(df_user_activity["user_activity_day"] > trainSpan[1]) & (
        df_user_activity["user_activity_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user = list(set(active_app_launch + active_video_create + active_user_activity))
        del df_user_register, df_app_launch, df_video_create, df_user_activity
        gc.collect()
    else:
        del df_user_register, df_app_launch, df_video_create, df_user_activity
        gc.collect()

    df_user_activity_train["user_user_activity_register_time"] = (
    df_user_activity_train["register_day"].apply(lambda x: (trainSpan[1] - x + 1))).astype(np.uint8)

    df_user_activity_train["user_user_activity_rate"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                             "user_activity_day"].transform("count")).astype(np.uint16)
    df_user_activity_train["user_user_activity_rate_spanInv"] = (
    df_user_activity_train["user_user_activity_rate"] * 1.0 / (
        trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_user_activity_train["user_user_activity_rate_resInv"] = (
    df_user_activity_train["user_user_activity_rate"] * 1.0 / \
    df_user_activity_train["user_user_activity_register_time"]).astype(np.float32)
    df_user_activity_train["user_user_activity_rate_spanResInv"] = (df_user_activity_train[
                                                                        "user_user_activity_rate_spanInv"] * 1.0 / \
                                                                    df_user_activity_train[
                                                                        "user_user_activity_register_time"]).astype(
        np.float32)

    user_activity_author = df_user_activity_train["author_id"].unique().tolist()
    user_authors = df_user_activity_train.loc[
        df_user_activity_train["user_id"].isin(user_activity_author), "author_id"].tolist()
    user_author_count_dict = Counter(list(user_authors))
    df_user_activity_train["user_in_author_rate"] = df_user_activity_train["user_id"].apply(
        lambda x: 0 if x not in user_author_count_dict.keys() else user_author_count_dict.get(x)).astype(np.uint16)
    print("begin to get the complete forward feature ")
    user_activity_rate_name_forward_ls = []
    for rf in range(1, 7, 1):
        rate_name_forward = "user_user_activity_rate_f" + str(rf)
        user_activity_rate_name_forward_ls.append(rate_name_forward)
        df_user_activity_train[rate_name_forward] = (df_user_activity_train.groupby(by=["user_id"])[
            "user_activity_day"].transform(
            lambda x: count_occurence(x, (trainSpan[0], trainSpan[1] - rf)))).astype(np.uint16)
        rate_name_forward_spanInv = rate_name_forward + "_spanInv"
        user_activity_rate_name_forward_ls.append(rate_name_forward_spanInv)
        df_user_activity_train[rate_name_forward_spanInv] = (df_user_activity_train[rate_name_forward] * 1.0 / (
            trainSpan[1] - trainSpan[0] - rf)).astype(np.float32)
        rate_name_forward_resInv = rate_name_forward + "_resInv"
        user_activity_rate_name_forward_ls.append(rate_name_forward_resInv)
        df_user_activity_train[rate_name_forward_resInv] = (df_user_activity_train[rate_name_forward] * 1.0 / \
                                                            df_user_activity_train[
                                                                "user_user_activity_register_time"]).astype(np.float32)
        rate_name_forward_spanResInv = rate_name_forward + "_spanResInv"
        user_activity_rate_name_forward_ls.append(rate_name_forward_spanResInv)
        df_user_activity_train[rate_name_forward_spanResInv] = (
        df_user_activity_train[rate_name_forward_spanInv] * 1.0 / \
        df_user_activity_train["user_user_activity_register_time"]).astype(np.float32)
    user_activity_rate_name_backward_ls = []
    print("begin to get the complete backward feature ")
    for rb in range(14, -1, -1):
        rate_name_backward = "user_user_activity_rate_b" + str(rb)
        user_activity_rate_name_backward_ls.append(rate_name_backward)
        df_user_activity_train[rate_name_backward] = (df_user_activity_train.groupby(by=["user_id"])[
            "user_activity_day"].transform(
            lambda x: count_occurence(x, (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint16)
        rate_name_backward_resInv = rate_name_backward + "_resInv"
        user_activity_rate_name_backward_ls.append(rate_name_backward_resInv)

        df_user_activity_train[rate_name_backward_resInv] = (df_user_activity_train[rate_name_backward] * 1.0 / \
                                                             df_user_activity_train[
                                                                 "user_user_activity_register_time"]).astype(np.float32)
    # df_user_activity_train.drop_duplicates(inplace=True)
    print("begin to get the page ratio ")
    user_page_name_ls = []
    for i in range(0, 5):
        page_name = "user_page_ratio" + str(i)
        user_page_name_ls.append(page_name)
        df_user_activity_train[page_name] = (
        df_user_activity_train.groupby(by=["user_id"])["page"].transform(lambda x: get_ratio(list(x), i)) * 1.0).astype(
            np.float32)
    df_user_activity_train["user_action_type_num"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                          "action_type"].transform(lambda x: x.nunique())).astype(
        np.uint8)
    user_action_type_name_ls = []
    print("begin to get the action type ratio ")
    for i in range(0, 6):
        action_type_name = "user_action_type_ratio" + str(i)
        user_action_type_name_ls.append(action_type_name)
        df_user_activity_train[action_type_name] = (
        df_user_activity_train.groupby(by=["user_id"])["action_type"].transform(
            lambda x: get_ratio(list(x), i)) * 1.0).astype(np.float32)

    df_user_activity_train.drop(labels=["page", "action_type"], axis=1, inplace=True)
    df_user_activity_train["user_user_activity_var"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                            "user_activity_day"].transform(
        lambda x: np.var(list(x))) * 1.0).astype(np.float32)
    user_activity_var_name_ls = []
    print("begin to get the var ")
    for v in range(5, 12, 3):
        var_name = "user_user_activity_var_b" + str(v)
        df_user_activity_train[var_name] = (df_user_activity_train.groupby(by=["user_id"])[
                                                "user_activity_day"].transform(
            lambda x: get_var(list(x), trainSpan[1], v)) * 1.0).astype(np.float32)
        user_activity_var_name_ls.append(var_name)

    df_user_activity_train.drop_duplicates(inplace=True)

    df_user_activity_train["user_user_activity_day"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                            "user_activity_day"].transform(
        lambda x: x.nunique())).astype(
        np.uint8)
    df_user_activity_train["user_user_activity_day_spanInv"] = (
    df_user_activity_train["user_user_activity_day"] * 1.0 / (
        trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_user_activity_train["user_user_activity_day_resInv"] = (df_user_activity_train["user_user_activity_day"] * 1.0 / \
                                                               df_user_activity_train[
                                                                   "user_user_activity_register_time"]).astype(
        np.float32)
    df_user_activity_train["user_user_activity_day_spanResInv"] = (
    df_user_activity_train["user_user_activity_day_spanInv"] * 1.0 / \
    df_user_activity_train["user_user_activity_register_time"]).astype(np.float32)
    df_user_activity_train["user_user_activity_frequency"] = (df_user_activity_train["user_user_activity_rate"] * 1.0 / \
                                                              df_user_activity_train["user_user_activity_day"]).astype(
        np.float32)
    df_user_activity_train["user_video_rate"] = (df_user_activity_train.groupby(by=["user_id"])["video_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity_train["user_video_rate_spanInv"] = (
    df_user_activity_train["user_video_rate"] * 1.0 / (trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_user_activity_train["user_video_rate_resInv"] = (
    df_user_activity_train["user_video_rate"] * 1.0 / df_user_activity_train[
        "user_user_activity_register_time"]).astype(np.float32)
    df_user_activity_train["user_video_rate_spanResInv"] = (
    df_user_activity_train["user_video_rate_spanInv"] * 1.0 / df_user_activity_train[
        "user_user_activity_register_time"]).astype(np.float32)

    df_user_activity_train["user_author_rate"] = (df_user_activity_train.groupby(by=["user_id"])["author_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity_train["user_author_rate_spanInv"] = (
    df_user_activity_train["user_author_rate"] * 1.0 / (trainSpan[1] - trainSpan[0] + 1)).astype(np.float32)
    df_user_activity_train["user_author_rate_resInv"] = (
    df_user_activity_train["user_author_rate"] * 1.0 / df_user_activity_train[
        "user_user_activity_register_time"]).astype(np.float32)
    df_user_activity_train["user_author_rate_spanResInv"] = (
    df_user_activity_train["user_author_rate_spanInv"] * 1.0 / df_user_activity_train[
        "user_user_activity_register_time"]).astype(np.float32)
    df_user_activity_train["user_user_activity_last_time"] = (
    df_user_activity_train.groupby(by=["user_id"])["user_activity_day"].transform(
        lambda x: trainSpan[1] - max(x))).astype(np.uint8)
    # user_activity_feature_temp = [
    #                          "user_user_activity_rate",
    #                          "user_user_activity_rate_spanInv",
    #                          "user_user_activity_rate_resInv",
    #                          "user_user_activity_rate_spanResInv",
    #                          "user_user_activity_day",
    #                          "user_user_activity_day_spanInv",
    #                          "user_user_activity_day_resInv",
    #                          "user_user_activity_day_spanResInv",
    #                          "user_user_activity_frequency",
    #                          "user_video_rate",
    #                          "user_video_rate_spanInv",
    #                          "user_video_rate_resInv",
    #                          "user_video_rate_spanResInv",
    #                          "user_author_rate",
    #                          "user_author_rate_spanInv",
    #                          "user_author_rate_resInv",
    #                          "user_author_rate_spanResInv",
    #                          "user_user_activity_var",
    #                          "user_action_type_num",
    #                          "user_user_activity_last_time",
    #                          ]\
    #                        + user_activity_var_name_ls\
    #                         +user_page_name_ls\
    #                         +user_action_type_name_ls
    # df_user_activity_train[user_activity_feature_temp].to_csv("data/user_activity_temp.csv",header=True,index=False)
    # df_user_activity_train.drop(labels=user_activity_feature_temp,axis=1,inplace=True)
    df_user_activity_train.drop(labels=["video_id", "author_id"], axis=1, inplace=True)
    df_user_activity_train.drop_duplicates(inplace=True)
    print("begin to get the reduced forward feature ")
    for rf in range(1, 7, 1):
        rate_name_forward = "user_user_activity_rate_f" + str(rf)
        day_name_forward = "user_user_activity_day_f" + str(rf)
        frequency_name_forward = "user_user_activity_frequency_f" + str(rf)
        user_activity_rate_name_forward_ls.append(day_name_forward)
        user_activity_rate_name_forward_ls.append(frequency_name_forward)
        df_user_activity_train[day_name_forward] = (df_user_activity_train.groupby(by=["user_id"])[
            "user_activity_day"].transform(
            lambda x: count_occurence(set(x), (trainSpan[0], trainSpan[1] - rf)))).astype(np.uint8)
        df_user_activity_train[frequency_name_forward] = (
        df_user_activity_train[rate_name_forward] * 1.0 / df_user_activity_train[day_name_forward]).astype(np.float32)

        rate_name_forward_spanInv = rate_name_forward + "_spanInv"
        day_name_forward_spanInv = day_name_forward + "_spanInv"
        user_activity_rate_name_forward_ls.append(day_name_forward_spanInv)
        df_user_activity_train[day_name_forward_spanInv] = (df_user_activity_train[day_name_forward] * 1.0 / (
            trainSpan[1] - trainSpan[0] - rf)).astype(np.float32)
        rate_name_forward_resInv = rate_name_forward + "_resInv"
        day_name_forward_resInv = day_name_forward + "_resInv"
        user_activity_rate_name_forward_ls.append(day_name_forward_resInv)
        df_user_activity_train[day_name_forward_resInv] = (df_user_activity_train[day_name_forward] * 1.0 / \
                                                           df_user_activity_train[
                                                               "user_user_activity_register_time"]).astype(np.float32)

        rate_name_forward_spanResInv = rate_name_forward + "_spanResInv"
        day_name_forward_spanResInv = day_name_forward + "_spanResInv"
        user_activity_rate_name_forward_ls.append(day_name_forward_spanResInv)
        df_user_activity_train[day_name_forward_spanResInv] = (df_user_activity_train[day_name_forward_spanInv] * 1.0 / \
                                                               df_user_activity_train[
                                                                   "user_user_activity_register_time"]).astype(
            np.float32)
    print("begin to get the reduced backward feature ")
    for rb in range(14, -1, -1):
        rate_name_backward = "user_user_activity_rate_b" + str(rb)
        day_name_backward = "user_user_activity_day_b" + str(rb)
        user_activity_rate_name_backward_ls.append(day_name_backward)
        if rb != 0:
            frequency_name_backward = "user_user_activity_frequency_b" + str(rb)
            user_activity_rate_name_backward_ls.append(frequency_name_backward)

        df_user_activity_train[day_name_backward] = (df_user_activity_train.groupby(by=["user_id"])[
            "user_activity_day"].transform(
            lambda x: count_occurence(set(x), (trainSpan[1] - rb, trainSpan[1] + 1)))).astype(np.uint8)
        if rb != 0:
            df_user_activity_train[frequency_name_backward] = (
            df_user_activity_train[rate_name_backward] * 1.0 / df_user_activity_train[day_name_backward]).astype(
                np.float32)
        rate_name_backward_resInv = rate_name_backward + "_resInv"
        day_name_backward_resInv = day_name_backward + "_resInv"
        user_activity_rate_name_backward_ls.append(day_name_backward_resInv)

        df_user_activity_train[day_name_backward_resInv] = (df_user_activity_train[day_name_backward] * 1.0 / \
                                                            df_user_activity_train[
                                                                "user_user_activity_register_time"]).astype(np.float32)

    df_user_activity_train["user_user_activity_gap"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                            "user_activity_day"].transform(
        lambda x: (len(set(x)) - 1) * 1.0 / (max(x) - min(x)) if len(set(x)) > 1 else min(x) * 1.0 / (
        trainSpan[1] + min(x))) * 1.0).astype(np.float32)
    user_activity_gap_name_ls = []
    for g in range(5, 12, 3):
        gap_name = "user_user_activity_gap_b" + str(g)
        df_user_activity_train[gap_name] = (df_user_activity_train.groupby(by=["user_id"])[
                                                "user_activity_day"].transform(
            lambda x: get_gap(x, trainSpan[1], g)) * 1.0).astype(np.float32)
        user_activity_gap_name_ls.append(gap_name)

    df_user_activity_train["user_user_activity_day_var"] = (df_user_activity_train.groupby(by=["user_id"])[
                                                                "user_activity_day"].transform(
        lambda x: np.var(list(set(x)))) * 1.0).astype(np.float32)
    user_activity_day_var_name_ls = []
    for v in range(5, 12, 3):
        day_var_name = "user_user_activity_day_var_b" + str(v)
        df_user_activity_train[day_var_name] = (df_user_activity_train.groupby(by=["user_id"])[
                                                    "user_activity_day"].transform(
            lambda x: get_var(list(set(x)), trainSpan[1], v)) * 1.0).astype(np.float32)
        user_activity_day_var_name_ls.append(day_var_name)
    # df_user_activity_train_temp = pd.read_csv("data/user_activity_temp.csv",header=0,index_col=None)
    # df_user_activity_train = pd.concat([df_user_activity_train,df_user_activity_train_temp],axis=1)
    df_user_activity_train.drop_duplicates(inplace=True)

    user_activity_feature = ["user_id",
                             "user_user_activity_rate",
                             "user_user_activity_rate_spanInv",
                             "user_user_activity_rate_resInv",
                             "user_user_activity_rate_spanResInv",
                             "user_user_activity_day",
                             "user_user_activity_day_spanInv",
                             "user_user_activity_day_resInv",
                             "user_user_activity_day_spanResInv",
                             "user_user_activity_frequency",
                             "user_in_author_rate",
                             "user_video_rate",
                             "user_video_rate_spanInv",
                             "user_video_rate_resInv",
                             "user_video_rate_spanResInv",
                             "user_author_rate",
                             "user_author_rate_spanInv",
                             "user_author_rate_resInv",
                             "user_author_rate_spanResInv",
                             "user_user_activity_gap",
                             "user_user_activity_day_var",
                             "user_user_activity_var",
                             "user_action_type_num",
                             "user_user_activity_last_time",
                             ] \
                            + user_activity_rate_name_forward_ls \
                            + user_activity_rate_name_backward_ls \
                            + user_activity_gap_name_ls \
                            + user_activity_var_name_ls \
                            + user_activity_day_var_name_ls \
                            + user_page_name_ls \
                            + user_action_type_name_ls

    df_user_activity_train = df_user_activity_train[user_activity_feature].drop_duplicates()
    # df_user_activity_train = df_user_activity_train[user_activity_feature]
    ds4 = df_user_activity_train.describe()
    print(ds4)
    ds4.to_csv("kuaishou_stats2.csv", mode='a')
    # df_user_register_train = pd.read_csv("data/user_register_temp.csv",header=0,index_col=None)
    # df_app_launch_train = pd.read_csv("data/user_app_launch_temp.csv",header=0,index_col=None)
    # df_video_create_train = pd.read_csv("data/user_video_create_temp.csv",header=0,index_col=None)
    if label:
        df_user_register_train["label"] = 0
        df_user_register_train.loc[df_user_register_train["user_id"].isin(active_user), "label"] = 1

        df_app_launch_train["label"] = 0
        df_app_launch_train.loc[df_app_launch_train["user_id"].isin(active_user), "label"] = 1

        df_video_create_train["label"] = 0
        df_video_create_train.loc[df_video_create_train["user_id"].isin(active_user), "label"] = 1

        df_user_activity_train["label"] = 0
        df_user_activity_train.loc[df_user_activity_train["user_id"].isin(active_user), "label"] = 1
    df_launch_register = df_app_launch_train.merge(df_user_register_train, how="left").fillna(0)
    # print(df_register_launch.describe())
    df_launch_register_create = df_launch_register.merge(df_video_create_train, how="left").fillna(0)
    # print(df_register_launch_create.describe())
    # df_activity_register_launch_create = df_user_activity_train.merge(df_launch_register_create,how="left").fillna(0)
    df_launch_activity_register_create = df_launch_register_create.merge(df_user_activity_train, how="left").fillna(0)
    print("before drop the duplicates of user activity log")
    print(df_launch_activity_register_create.describe())
    keep_feature = list(set(user_register_feature + app_launch_feature + video_create_feature + user_activity_feature))
    df_launch_activity_register_create.drop_duplicates(subset=keep_feature, inplace=True)
    print("after drop the duplicates of user activity log")
    ds5 = df_launch_activity_register_create.describe()
    print(ds5)
    ds5.to_csv("kuaishou_stats2.csv", mode='a')
    return df_launch_activity_register_create

# if __name__=="__main__":
#     train_set = processing((1,15),label=True)
#     print(train_set.info())
