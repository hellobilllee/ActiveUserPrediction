import csv
import datetime
import pandas as pd
import numpy as np
user_register_log = ["user_id","register_day","register_type","device_type"]
app_launch_log = ["user_id","app_launch_day"]
video_create_log = ["user_id","video_create_day"]
user_activity_log = ["user_id","user_activity_day","page","video_id","author_id","action_type"]


def get_user_from_videoCreate(laterThanDay,videoCount):
    print("get users from video create")
    video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_table("data/video_create_log.txt",header=None,names=video_create_log,index_col=None,dtype=dtype_video_create)
    latest_user = (df_video_create.loc[df_video_create["video_create_day"]>laterThanDay]).user_id.unique().tolist()
    print("get latest users")
    print(latest_user)
    print(len(latest_user))
    df_video_create["videoCount"] = df_video_create.groupby(by=["user_id"])["video_create_day"].transform(lambda x: x.nunique())
    frequent_user = (df_video_create.loc[df_video_create["videoCount"]>videoCount]).user_id.unique().tolist()
    print("get frequent users")
    print(frequent_user)
    print(len(frequent_user))
    user_videoCreate = list(set(latest_user+frequent_user))
    print(user_videoCreate)
    print(len(user_videoCreate))
    return user_videoCreate
    # with open("result/submission.csv","a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in user_videoCreate:
    #         writer.writerow([i])
# get_user_from_videoCreate(23,2)
def get_user_from_appLaunch(laterThanDay,launchCount):
    print("get users from app launch log")
    app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch =  {"user_id":np.uint32,"app_launch_day":np.uint8}
    df_app_launch = pd.read_table("data/app_launch_log.txt",header=None,names=app_launch_log,index_col=None,dtype=dtype_app_launch)
    latest_user = (df_app_launch.loc[df_app_launch["app_launch_day"]>laterThanDay]).user_id.unique().tolist()
    print("get latest users")
    print(latest_user)
    print(len(latest_user))
    df_app_launch["launchCount"] = df_app_launch.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: x.nunique())
    frequent_user = (df_app_launch.loc[df_app_launch["launchCount"]>launchCount]).user_id.unique().tolist()
    print("get frequent users")
    print(frequent_user)
    print(len(frequent_user))
    user_appLaunch = list(set(latest_user+frequent_user))
    print("get merged users")
    print(user_appLaunch)
    print(len(user_appLaunch))
    return user_appLaunch
    # with open("result/submission.csv","a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in user_appLaunch:
    #         writer.writerow([i])
# get_user_from_appLaunch(27,4)
def get_user_from_userRegister(laterThanDay):
    print("get users from user register log")
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": str}
    df_user_register = pd.read_table("data/user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register)
    latest_user = (df_user_register.loc[df_user_register["register_day"]>laterThanDay]).user_id.unique().tolist()
    print("get latest users")
    print(latest_user)
    print(len(latest_user))
    return latest_user
# get_user_from_userRegister(25)
def get_user_from_userActivity(laterThanDay,dayCount,pageList,typeList):
    print("get users from user activity log")
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    usecols = ["user_id", "user_activity_day", "page","action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "action_type": np.uint8}
    df_user_activity = pd.read_table("data/user_activity_log.txt",header=None,names=user_activity_log,usecols=usecols,index_col=None,dtype=dtype_user_activity)
    latest_user = (df_user_activity.loc[df_user_activity["user_activity_day"]>laterThanDay]).user_id.unique().tolist()
    print("get latest users")
    print(latest_user)
    print(len(latest_user))

    df_user_activity["dayCount"] = df_user_activity.groupby(by=["user_id"])["user_activity_day"].transform(lambda x: x.nunique())
    frequent_user = (df_user_activity.loc[df_user_activity["dayCount"]>dayCount]).user_id.unique().tolist()
    print("get frequent users")
    print(frequent_user)
    print(len(frequent_user))

    print("get users in certain pages and certain action type")
    user_inList = (df_user_activity.loc[((df_user_activity["page"].isin(pageList))|(df_user_activity["action_type"].isin(typeList)))&(df_user_activity["user_activity_day"]>laterThanDay-3)]).user_id.unique().tolist()

    print(user_inList)
    print(len(user_inList))
    user_userActivity = list(set(latest_user+frequent_user+user_inList))

    print("get merged users")
    print(user_userActivity)
    print(len(user_userActivity))
    return user_userActivity
# get_user_from_userActivity(27, 3, [1,2,3], [1,3,4,5])

def get_user():

    user_videoCreate = get_user_from_videoCreate(23, 31)
    user_appLaunch = get_user_from_appLaunch(23,31)
    user_userRegister = get_user_from_userRegister(23)
    user_userActivity = get_user_from_userActivity(23, 31, [], [])

    users = list(set(user_videoCreate+user_appLaunch+user_userRegister+user_userActivity))
    print("get the final merged users")
    print(users)
    print(len(users))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_" + str_time + ".csv"
    # with open(submission_file,"a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in users:
    #         writer.writerow([i])
get_user()