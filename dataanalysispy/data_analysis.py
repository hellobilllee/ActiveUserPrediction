import pandas as pd
import numpy as np
from sklearn import preprocessing

from data_process_v6 import processing


def user_register():
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": str}
    df_user_register = pd.read_table("data/user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register)
    le = preprocessing.LabelEncoder()
    df_user_register["device_type"] = pd.Series(le.fit_transform(df_user_register["device_type"].values)).astype(np.uint16)
    des_user_register= df_user_register.describe(include="all")
    print(des_user_register)
    des_user_register.to_csv("kuaishou_stats.csv", mode="a")
    df_user_register.to_csv("data/user_register_log.csv",header=True,index=False)

def app_launch():
    app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch =  {"user_id":np.uint32,"app_launch_day":np.uint8}
    df_app_launch = pd.read_table("data/app_launch_log.txt",header=None,names=app_launch_log,index_col=None,dtype=dtype_app_launch)
    des_app_launch= df_app_launch.describe(include="all")
    print(des_app_launch)
    # des_app_launch.to_csv("kuaishou_stats.csv", mode="a")
    df_app_launch.to_csv("data/app_launch_log.csv",header=True,index=False)
# app_launch()
def video_create():
    video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_table("data/video_create_log.txt",header=None,names=video_create_log,index_col=None,dtype=dtype_video_create)
    des_video_create= df_video_create.describe(include="all")
    print(des_video_create)
    # des_video_create.to_csv("kuaishou_stats.csv", mode="a")
    df_video_create.to_csv("data/video_create_log.csv",header=True,index=False)
# video_create()
def user_activity():
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id":np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_table("data/user_activity_log.txt",header=None,names=user_activity_log,index_col=None,dtype=dtype_user_activity)
    # le = preprocessing.LabelEncoder()
    # for fea in ["user_activity_day", "page", "action_type"]:
    #     df_user_activity[fea] = pd.Series(le.fit_transform(df_user_activity[fea].values)).astype(np.uint8)
    des_user_activity= df_user_activity.describe(include="all")
    print(des_user_activity)
    des_user_activity.to_csv("kuaishou_stats.csv", mode="a")
    df_user_activity.to_csv("data/user_activity_log.csv",header=True,index=False)
def mergeRegister():
    print("get users from user register log")
    # user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type":np.uint16}
    df_user_register_train = pd.read_csv("data/user_register_log.csv",header=0,index_col=None,dtype=dtype_user_register)
    # df_user_register.drop_duplicates(inplace=True)
    # df_user_register_train = df_user_register.loc[(df_user_register["register_day"]>=trainSpan[0])&(df_user_register["register_day"]<=trainSpan[1])]

    df_user_register_train["register_day_rate"] = df_user_register_train.groupby(by=["register_day"])["register_day"].transform("count")
    df_user_register_train["register_type_rate"] = df_user_register_train.groupby(by=["register_type"])["register_type"].transform("count")
    df_user_register_train["register_type_device"] = df_user_register_train.groupby(by=["register_type"])["device_type"].transform(lambda x: x.nunique())
    df_user_register_train["device_type_rate"] = df_user_register_train.groupby(by=["device_type"])["device_type"].transform("count")
    df_user_register_train["device_type_register"] = df_user_register_train.groupby(by=["device_type"])["register_type"].transform(lambda x: x.nunique())

    df_user_register = df_user_register_train.drop(labels=["register_type","device_type"],axis=1)

    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {"user_id": np.uint32, "app_launch_day": np.uint8}
    df_app_launch = pd.read_csv("data/app_launch_log.csv", header=0, index_col=None, dtype=dtype_app_launch)
def analysisTrans():
    print("begin to load the trainset1")
    # train_set1 = processing(trainSpan=(1,10),label=True)
    # train_set1.to_csv("data/training_ld1-10.csv", header=True, index=False)
    train_set1 = pd.read_csv("data/training_ld1-10.csv", header=0, index_col=None)
    print(train_set1.describe())
    print("begin to load the trainset2")
    # train_set2 = processing(trainSpan=(11,20),label=True)
    # train_set2.to_csv("data/training_ld11-20.csv", header=True, index=False)
    train_set2 = pd.read_csv("data/training_ld11-20.csv", header=0, index_col=None)
    print(train_set2.describe())
    print("begin to merge the trainsets")
    train_set = pd.concat([train_set1,train_set2],axis=0)
    print(train_set.describe())
analysisTrans()
# user_activity()







