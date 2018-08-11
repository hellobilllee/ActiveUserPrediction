import pandas as pd
import numpy as np
user_register_log = ["user_id","register_day","register_type","device_type"]
app_launch_log = ["user_id","app_launch_day"]
video_create_log = ["user_id","video_create_day"]
user_activity_log = ["user_id","user_activity_day","page","video_id","author_id","action_type"]

user_register_feature = ["user_id",
                         "register_day_rate","register_type_rate",
                         "register_type_device","device_type_rate","device_type_register"]
app_launch_feature = ["user_id",
                      "user_app_launch_rate","user_app_launch_gap"]
video_create_feature = ["user_id",
                        "user_video_create_rate","user_video_create_day","user_video_create_gap"]
user_activity_feature = ["user_id",
                         "user_activity_day","user_activity_day_rate","user_activity_gap",
                         # "page_rate","page_action_type",
                         # "video_id_rate","video_id_user","video_id_action_type",
                         # "author_id_rate","author_id_user","author_id_video",
                         # "action_type_rate","action_type_page"
                         ]

def processing(trainSpan=(1,23),label=True):
    if label:
        assert isinstance(trainSpan,tuple),"input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0]>0 and trainSpan[0]<23 and trainSpan[1]>trainSpan[0] and trainSpan[1]<=23
    else:
        assert isinstance(trainSpan,tuple),"input parameter should be a tuple with two items (min,max)"
        assert trainSpan[0]>0 and trainSpan[0]<30 and trainSpan[1]>trainSpan[0] and trainSpan[1]<=30
    print("get users from user register log")
    # user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type":np.uint16}
    df_user_register = pd.read_csv("data/user_register_log.csv",header=0,index_col=None,dtype=dtype_user_register)
    # df_user_register.drop_duplicates(inplace=True)
    df_user_register_train = df_user_register.loc[(df_user_register["register_day"]>=trainSpan[0])&(df_user_register["register_day"]<=trainSpan[1])]

    df_user_register_train["register_day_rate"] = df_user_register_train.groupby(by=["register_day"])["register_day"].transform("count")
    df_user_register_train["register_type_rate"] = df_user_register_train.groupby(by=["register_type"])["register_type"].transform("count")
    df_user_register_train["register_type_device"] = df_user_register_train.groupby(by=["register_type"])["device_type"].transform(lambda x: x.nunique())
    df_user_register_train["device_type_rate"] = df_user_register_train.groupby(by=["device_type"])["device_type"].transform("count")
    df_user_register_train["device_type_register"] = df_user_register_train.groupby(by=["device_type"])["register_type"].transform(lambda x: x.nunique())

    df_user_register_train = df_user_register_train[user_register_feature].drop_duplicates()
    print(df_user_register_train.describe())

    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {"user_id": np.uint32, "app_launch_day": np.uint8}
    df_app_launch = pd.read_csv("data/app_launch_log.csv", header=0, index_col=None, dtype=dtype_app_launch)
    # df_app_launch.drop_duplicates(inplace=True)
    df_app_launch_train = df_app_launch.loc[
        (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]

    # print(df_app_launch_train.describe())
    df_app_launch_train["user_app_launch_rate"] = df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform("count")
    df_app_launch_train["user_app_launch_gap"] = df_app_launch_train.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: (max(x) - min(x)) / (len(x) - 1) if len(set(x)) > 1 else 0)

    df_app_launch_train = df_app_launch_train[app_launch_feature].drop_duplicates()
    print(df_app_launch_train.describe())

    print("get users from video create")
    # video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_csv("data/video_create_log.csv",header=0,index_col=None,dtype=dtype_video_create)
    # df_video_create.drop_duplicates(inplace=True)
    df_video_create_train = df_video_create.loc[
        (df_video_create["video_create_day"] >= trainSpan[0]) & (df_video_create["video_create_day"] <= trainSpan[1])]

    df_video_create_train["user_video_create_rate"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform("count")
    df_video_create_train["user_video_create_day"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: x.nunique())
    df_video_create_train["user_video_create_gap"] = df_video_create_train.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    # print(df_video_create_train.describe())
    df_video_create_train = df_video_create_train[video_create_feature].drop_duplicates()
    print(df_video_create_train.describe())

    print("get users from user activity log")
    # user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    # usecols = ["user_id", "user_activity_day", "page","action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id": np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_csv("data/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity)
    # print(df_user_activity.describe())
    # df_user_activity.drop_duplicates(inplace=True)
    # print(df_user_activity.describe())
    df_user_activity_train = df_user_activity.loc[
        (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
        df_user_activity["user_activity_day"] <= trainSpan[1])]

    df_user_activity_train["user_activity_rate"] = df_user_activity_train.groupby(by=["user_id"])["user_id"].transform(
        "count")
    df_user_activity_train["user_activity_day_rate"] = df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: x.nunique())
    df_user_activity_train["user_activity_gap"] = df_user_activity_train.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x))>1 else 0)
    # df_user_activity_train["page_rate"] = df_user_activity_train.groupby(by=["page"])["page"].transform("count")
    # df_user_activity_train["page_action_type"] = df_user_activity_train.groupby(by=["page"])["action_type"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["video_id_rate"] = df_user_activity_train.groupby(by=["video_id"])["video_id"].transform(
    #     "count")
    # df_user_activity_train["video_id_user"] = df_user_activity_train.groupby(by=["video_id"])["user_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["video_id_action_type"] = df_user_activity_train.groupby(by=["video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())
    # df_user_activity_train["author_id_rate"] = df_user_activity_train.groupby(by=["author_id"])["author_id"].transform(
    #     "count")
    # df_user_activity_train["author_id_user"] = df_user_activity_train.groupby(by=["author_id"])["user_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["author_id_video"] = df_user_activity_train.groupby(by=["author_id"])["video_id"].transform(
    #     lambda x: x.nunique())
    # df_user_activity_train["action_type_rate"] = df_user_activity_train.groupby(by=["action_type"])[
    #     "action_type"].transform("count")
    # df_user_activity_train["action_type_page"] = df_user_activity_train.groupby(by=["action_type"])["page"].transform(
    #     lambda x: x.nunique())
    df_user_activity_train = df_user_activity_train[user_activity_feature].drop_duplicates()
    print(df_user_activity_train.describe())

    if label:
        active_user_register = (df_user_register.loc[(df_user_register["register_day"]>trainSpan[1])&(df_user_register["register_day"]<=(trainSpan[1]+7))]).user_id.unique().tolist()
        active_app_launch = (df_app_launch.loc[(df_app_launch["app_launch_day"] > trainSpan[1]) & (df_app_launch["app_launch_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_video_create = (df_video_create.loc[(df_video_create["video_create_day"]>trainSpan[1])&(df_video_create["video_create_day"]<=(trainSpan[1]+7))]).user_id.unique().tolist()
        active_user_activity = (df_user_activity.loc[(df_user_activity["user_activity_day"] > trainSpan[1]) & (df_user_activity["user_activity_day"] <= (trainSpan[1] + 7))]).user_id.unique().tolist()
        active_user = list(set(active_user_register+active_app_launch+active_video_create+active_user_activity))

        df_user_register_train["label"] = 0
        df_user_register_train.loc[df_user_register_train["user_id"].isin(active_user),"label"] = 1

        df_app_launch_train["label"] = 0
        df_app_launch_train.loc[df_app_launch_train["user_id"].isin(active_user),"label"] = 1

        df_video_create_train["label"] = 0
        df_video_create_train.loc[df_video_create_train["user_id"].isin(active_user),"label"] = 1

        df_user_activity_train["label"] = 0
        df_user_activity_train.loc[df_user_activity_train["user_id"].isin(active_user),"label"] = 1

    df_register_launch = df_user_register_train.merge(df_app_launch_train,how="left")
    # print(df_register_launch.describe())
    df_register_launch_create = df_register_launch.merge(df_video_create_train,how="left")
    # print(df_register_launch_create.describe())
    df_register_launch_create = df_register_launch_create.fillna(0)
    df_activity_register_launch_create = df_user_activity_train.merge(df_register_launch_create,how="left")
    df_activity_register_launch_create = df_activity_register_launch_create.fillna(0)
    print(df_activity_register_launch_create.describe())
    return df_activity_register_launch_create
