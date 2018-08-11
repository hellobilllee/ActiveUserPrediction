import numpy as np
import pandas as pd

def processing(laterThanDay,launchCount,videoCount,activityCount):
    print("get users from user register log")
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": np.uint32}
    df_user_register = pd.read_table("user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register)
    user_outliers = df_user_register[(df_user_register["register_type"] == 3) & (
    (df_user_register["device_type"] == 1) | (df_user_register["device_type"] == 223) | (
    df_user_register["device_type"] == 83))]["user_id"].unique().tolist()
    df_user_register = df_user_register[~df_user_register["user_id"].isin(user_outliers)]
    df_user_register = df_user_register.loc[df_user_register["register_day"]>laterThanDay]


    print("get users from app launch log")
    app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch =  {"user_id":np.uint32,"app_launch_day":np.uint8}
    df_app_launch = pd.read_table("app_launch_log.txt",header=None,names=app_launch_log,index_col=None,dtype=dtype_app_launch)
    df_app_launch = df_app_launch[~df_app_launch["user_id"].isin(user_outliers)]
    df_app_launch = df_app_launch.loc[df_app_launch["app_launch_day"] >laterThanDay]

    df_app_launch["launchCount"] = df_app_launch.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: x.nunique())
    frequent_user1 = (df_app_launch.loc[df_app_launch["launchCount"]>launchCount]).user_id.unique().tolist()
    print("number of frequent launch users after {} is  {} ".format(laterThanDay,len(frequent_user1)))
    video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_table("video_create_log.txt",header=None,names=video_create_log,index_col=None,dtype=dtype_video_create)

    df_video_create = df_video_create[~df_video_create["user_id"].isin(user_outliers)]

    df_video_create = df_video_create.loc[df_video_create["video_create_day"] >laterThanDay]

    df_video_create["videoCount"] = df_video_create.groupby(by=["user_id"])["video_create_day"].transform(lambda x: x.nunique())
    frequent_user2 = (df_video_create.loc[df_video_create["videoCount"]>videoCount]).user_id.unique().tolist()
    print("number of frequent video create users after {} is  {} ".format(laterThanDay,len(frequent_user2)))

    print("get users from user activity log")
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id":np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_table("user_activity_log.txt",header=None,names=user_activity_log,index_col=None,dtype=dtype_user_activity,usecols=["user_id", "user_activity_day"])
    df_user_activity = df_user_activity[~df_user_activity["user_id"].isin(user_outliers)]
    df_user_activity= df_user_activity.loc[df_user_activity["user_activity_day"] >laterThanDay]
    df_user_activity["dayCount"] = df_user_activity.groupby(by=["user_id"])["user_activity_day"].transform(lambda x: x.nunique())
    frequent_user3 = (df_user_activity.loc[df_user_activity["dayCount"]>activityCount]).user_id.unique().tolist()
    print("number of frequent activity users after {} is  {} ".format(laterThanDay,len(frequent_user3)))


processing(24,4,4,4)
