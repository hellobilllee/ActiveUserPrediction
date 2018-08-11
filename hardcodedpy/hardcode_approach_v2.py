import csv
import datetime
import pandas as pd
import numpy as np
user_register_log = ["user_id","register_day","register_type","device_type"]
app_launch_log = ["user_id","app_launch_day"]
video_create_log = ["user_id","video_create_day"]
user_activity_log = ["user_id","user_activity_day","page","video_id","author_id","action_type"]


def get_frequser_from_videoCreate(videoCount):
    print("get users from video create")
    video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_table("data/video_create_log.txt",header=None,names=video_create_log,index_col=None,dtype=dtype_video_create)
    # latest_user = (df_video_create.loc[df_video_create["video_create_day"]>laterThanDay]).user_id.unique().tolist()
    # print("get latest users")
    # print(latest_user)
    # print(len(latest_user))
    df_video_create["videoCount"] = df_video_create.groupby(by=["user_id"])["video_create_day"].transform(lambda x: x.nunique())
    frequent_user = (df_video_create.loc[df_video_create["videoCount"]>videoCount]).user_id.unique().tolist()
    print(df_video_create.describe())
    print("get frequent users")
    print(frequent_user)
    print(len(frequent_user))
    return frequent_user
    # with open("result/submission.csv","a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in user_videoCreate:
    #         writer.writerow([i])
# get_frequser_from_videoCreate(3)
def get_frequser_from_appLaunch(launchCount):
    print("get users from app launch log")
    app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch =  {"user_id":np.uint32,"app_launch_day":np.uint8}
    df_app_launch = pd.read_table("data/app_launch_log.txt",header=None,names=app_launch_log,index_col=None,dtype=dtype_app_launch)
    # latest_user = (df_app_launch.loc[df_app_launch["app_launch_day"]>laterThanDay]).user_id.unique().tolist()
    # print("get latest users")
    # print(latest_user)
    # print(len(latest_user))
    df_app_launch["launchCount"] = df_app_launch.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: x.nunique())
    frequent_user = (df_app_launch.loc[df_app_launch["launchCount"]>launchCount]).user_id.unique().tolist()
    print(df_app_launch.describe())
    print("get frequent users")
    print(frequent_user)
    print(len(frequent_user))
    return frequent_user
    # with open("result/submission.csv","a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in user_appLaunch:
    #         writer.writerow([i])
# get_frequser_from_appLaunch(10)
def get_user_from_userRegister(laterThanDay):
    print("get users from user regiser log")
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": str}
    df_user_register = pd.read_table("data/user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register)
    latest_user = (df_user_register.loc[df_user_register["register_day"]>laterThanDay]).user_id.unique().tolist()
    print("get latest users")
    print(latest_user)
    print(len(latest_user))
    return latest_user
# get_user_from_userRegister(25)
def get_frequser_from_userActivity(dayCount):
    print("get users from user activity log")
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    usecols = ["user_id", "user_activity_day"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "action_type": np.uint8}
    df_user_activity = pd.read_table("data/user_activity_log.txt",header=None,names=user_activity_log,usecols=usecols,index_col=None,dtype=dtype_user_activity).drop_duplicates()
    # latest_user = (df_user_activity.loc[df_user_activity["user_activity_day"]>laterThanDay]).user_id.unique().tolist()
    # print("get latest users")
    # print(latest_user)
    # print(len(latest_user))

    df_user_activity["dayCount"] = df_user_activity.groupby(by=["user_id"])["user_activity_day"].transform(lambda x: x.nunique())
    frequent_user = (df_user_activity.loc[df_user_activity["dayCount"]>dayCount]).user_id.unique().tolist()
    print(df_user_activity.describe())
    print("get frequent users")
    print(frequent_user)
    print(len(frequent_user))

    # print("get users in certain pages and certain action type")
    # user_inList = (df_user_activity.loc[((df_user_activity["page"].isin(pageList))|(df_user_activity["action_type"].isin(typeList)))&(df_user_activity["user_activity_day"]>laterThanDay-3)]).user_id.unique().tolist()
    #
    # print(user_inList)
    # print(len(user_inList))
    # user_userActivity = list(set(latest_user+frequent_user+user_inList))
    #
    # print("get merged users")
    # print(user_userActivity)
    # print(len(user_userActivity))
    return frequent_user
# get_frequser_from_userActivity(10)
def get_activeUsers_from_register():
    print("get users from user regiser log")
    # user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": np.uint16}
    # df_user_register = pd.read_table("data/user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register)
    df_user_register = pd.read_csv("data/user_register_log.csv", header=0, index_col=None, dtype=dtype_user_register)

    df_user_register["register_type_rate"] = df_user_register.groupby(by=["register_type"])["register_type"].transform(
        "count")
    df_user_register["register_type_device"] = df_user_register.groupby(by=["register_type"])["device_type"].transform(
        lambda x: x.nunique())
    df_user_register["device_type_rate"] = df_user_register.groupby(by=["device_type"])["device_type"].transform(
        "count")
    df_user_register["device_type_register"] = df_user_register.groupby(by=["device_type"])["register_type"].transform(
        lambda x: x.nunique())
    active_users = pd.read_csv("hCoded/submission_freqUsers1_2018-06-08_11-16.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()

    df_acuser_info = df_user_register.loc[df_user_register["user_id"].isin(active_users)]
    # df_acuser_info.to_csv("data/active_user_info.csv",header=True,index=False)
    print(df_acuser_info.describe())
# get_activeUsers_from_register()
def get_user():

    user_videoCreate = get_frequser_from_videoCreate(3)
    user_appLaunch = get_frequser_from_appLaunch(8)
    # user_userRegister = get_user_from_userRegister(27)
    user_userActivity = get_frequser_from_userActivity(7)

    users = list(set(user_videoCreate+user_appLaunch+user_userActivity))
    print("get the final merged users")
    print(users)
    print(len(users))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "hCoded/submission_freqUsers_v2_" + str_time + ".csv"
    with open(submission_file,"a",newline="") as f:
        writer = csv.writer(f)
        for i in users:
            writer.writerow([i])
get_user()