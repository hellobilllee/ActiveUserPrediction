import csv
import datetime
import pandas as pd
import numpy as np
user_register_log = ["user_id","register_day","register_type","device_type"]
app_launch_log = ["user_id","app_launch_day"]
video_create_log = ["user_id","video_create_day"]
user_activity_log = ["user_id","user_activity_day","page","video_id","author_id","action_type"]


def get_user_from_videoCreate(laterThanDay,videoDayCount):
    print("get users from video create")
    video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_table("data/A2/video_create_log.txt",header=None,names=video_create_log,index_col=None,dtype=dtype_video_create).drop_duplicates()
    # print(df_video_create.groupby(by=["video_create_day"]).size())

    df_video_create["videoDayCount"] = df_video_create.groupby(by=["user_id"])["video_create_day"].transform(lambda x: x.nunique())
    user_videoCreate = (df_video_create.loc[(df_video_create["video_create_day"]>laterThanDay) & (df_video_create["videoDayCount"]>videoDayCount)]).user_id.unique().tolist()
    print("users created videos no later than {} and created video for more than {} days: {} ".format(laterThanDay,videoDayCount,len(user_videoCreate)))
    return user_videoCreate
    # with open("result/submission.csv","a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in user_videoCreate:
    #         writer.writerow([i])
# get_user_from_videoCreate(23,3)
def get_user_from_appLaunch(laterThanDay,launchCount):
    print("get users from app launch log")
    app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch =  {"user_id":np.uint32,"app_launch_day":np.uint8}
    df_app_launch = pd.read_table("data/A2/app_launch_log.txt",header=None,names=app_launch_log,index_col=None,dtype=dtype_app_launch).drop_duplicates()
    # print(df_app_launch.groupby(by=["user_id"]).size())
    # print(df_app_launch.groupby(by=["app_launch_day"]).size())
    df_app_launch["launchCount"] = df_app_launch.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: x.nunique())
    user_appLaunch = (df_app_launch.loc[(df_app_launch["app_launch_day"]>laterThanDay)&(df_app_launch["launchCount"]>launchCount)]).user_id.unique().tolist()
    print("users launched no later than {} and launched for more than {} days: {} ".format(laterThanDay,launchCount,len(user_appLaunch)))
    return user_appLaunch

    # with open("result/submission.csv","a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in user_appLaunch:
    #         writer.writerow([i])
# get_user_from_appLaunch(29,0)

def get_user_from_userRegister(laterThanDay):
    print("get users from user register log")
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": str}
    usecols = ["user_id", "register_day"]
    df_user_register = pd.read_table("data/A2/user_register_log.txt",header=None,names=user_register_log,index_col=None,dtype=dtype_user_register,usecols=usecols).drop_duplicates()
    # print(df_user_register.groupby(by=["register_day"]).size())
    print(df_user_register.groupby(by=["user_id","register_day"]).size())
    user_outliers = df_user_register[(df_user_register["register_day"]==24)&(df_user_register["register_type"]==3)&((df_user_register["device_type"]==1)|(df_user_register["device_type"]==223)|(df_user_register["device_type"]==83))]["user_id"].unique().tolist()
    df_user_register = df_user_register[~df_user_register["user_id"].isin(user_outliers)]
    latest_user = (df_user_register.loc[df_user_register["register_day"]>laterThanDay]).user_id.unique().tolist()
    # print("get latest users")
    print("get latest register users: {} ".format(len(latest_user)))
    # print(latest_user)
    # print(len(latest_user))
    return latest_user
get_user_from_userRegister(29)
def get_user_from_userActivity(laterThanDay,dayCount):
    print("get users from user activity log")
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    # usecols = ["user_id", "user_activity_day", "page","action_type"]
    usecols = ["user_id", "user_activity_day"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "action_type": np.uint8}
    df_user_activity = pd.read_table("data/A2/user_activity_log.txt",header=None,names=user_activity_log,usecols=usecols,index_col=None,dtype=dtype_user_activity).drop_duplicates()
    # print(df_user_activity[["user_id","user_activity_day"]].drop_duplicates().groupby(by=["user_activity_day"]).size())
    df_user_activity["dayCount"] = df_user_activity.groupby(by=["user_id"])["user_activity_day"].transform(lambda x: x.nunique())
    user_userActivity = (df_user_activity.loc[(df_user_activity["user_activity_day"]>laterThanDay)&(df_user_activity["dayCount"]>dayCount)]).user_id.unique().tolist()

    print("users activates no later than {} and activates for more than {} days: {} ".format(laterThanDay,dayCount ,len(user_userActivity)))
    return user_userActivity
def register_in_activity_author(laterThanDay,dayCount):
    print("get users from user activity log")
    user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    # usecols = ["user_id", "user_activity_day", "page","action_type"]
    usecols = ["user_id", "user_activity_day","author_id"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "action_type": np.uint8}
    df_user_activity = pd.read_table("data/user_activity_log.txt",header=None,names=user_activity_log,usecols=usecols,index_col=None,dtype=dtype_user_activity).drop_duplicates()
    df_user_activity["dayCount"] = df_user_activity.groupby(by=["user_id"])["user_activity_day"].transform(lambda x: x.nunique())
    author_id = df_user_activity["author_id"].unique().tolist()
    user_id = df_user_activity["user_id"].unique().tolist()

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    intersect_id = intersection(user_id, author_id)
    print("number of user is author {}".format(len(intersect_id)))
    user_userActivity = (df_user_activity.loc[(df_user_activity["user_activity_day"]>laterThanDay)&(df_user_activity["user_id"].isin(intersect_id))&(df_user_activity["dayCount"]>dayCount)]).user_id.unique().tolist()
    print("number of user is author activates more than {} days no later than {} : {}".format(dayCount,laterThanDay,len(user_userActivity)))
    return user_userActivity


# register_in_activity_author(23,2)
# users = get_user_from_userActivity(23,0)
# str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
# hdf = pd.Series(users,name="user_id")
# hfile = "hCoded/hcode_23_baseline_"+str_time + ".csv"
# hdf.to_csv(hfile,header=True,index=False)
def checkIn(ls1,ls2):
    if set(ls1)==set(ls2):
        print("the two lists are equal")
    elif set(ls1).issubset(set(ls2)):
        print("ls1 is subset of ls2")
    elif set(ls2).issubset(set(ls1)):
        print("ls2 is subset of ls1")
    else:
        print(" ls1 are different from ls2")
def get_user():

    # user_videoCreate13 = get_user_from_videoCreate(16,12)
    # user_videoCreate12 = get_user_from_videoCreate(17,11)
    # user_videoCreate11 = get_user_from_videoCreate(18,10)
    # user_videoCreate10 = get_user_from_videoCreate(19,9)
    # user_videoCreate9 = get_user_from_videoCreate(20,8)
    # user_videoCreate8 = get_user_from_videoCreate(21,7)
    # user_videoCreate7 = get_user_from_videoCreate(22,6)
    # user_videoCreate0 = get_user_from_videoCreate(23,5)
    # user_videoCreate1 = get_user_from_videoCreate(24,4)
    # user_videoCreate2 = get_user_from_videoCreate(25,4)
    user_videoCreate3 = get_user_from_videoCreate(26,3)
    user_videoCreate4 = get_user_from_videoCreate(27,2)
    user_videoCreate5 = get_user_from_videoCreate(28,1)
    user_videoCreate6 = get_user_from_videoCreate(29,0)
    # print("video creater {}".format(len(user_videoCreate0)))
    # print("video creater {}".format(len(user_videoCreate1)))
    # print("video creater {}".format(len(user_videoCreate2)))
    print("video creater {}".format(len(user_videoCreate3)))
    print("video creater {}".format(len(user_videoCreate4)))
    print("video creater {}".format(len(user_videoCreate5)))
    print("video creater {}".format(len(user_videoCreate6)))
    # print("video creater {}".format(len(user_videoCreate7)))
    # print("video creater {}".format(len(user_videoCreate8)))
    # print("video creater {}".format(len(user_videoCreate9)))
    # print("video creater {}".format(len(user_videoCreate10)))
    # print("video creater {}".format(len(user_videoCreate11)))
    # print("video creater {}".format(len(user_videoCreate12)))
    # print("video creater {}".format(len(user_videoCreate13)))
    user_videoCreate = list(set(
        # user_videoCreate0+
        # user_videoCreate1+
        # user_videoCreate2+
        user_videoCreate3+
        user_videoCreate4+
        user_videoCreate5+
        user_videoCreate6
        # user_videoCreate7+
        # user_videoCreate8+
        # user_videoCreate9+
        # user_videoCreate10+
        # user_videoCreate11+
        # user_videoCreate12+
        # user_videoCreate13
    ))
    print("video creater {}".format(len(user_videoCreate)))
    # user_appLaunch0 = get_user_from_appLaunch(23,6)
    # user_appLaunch1 = get_user_from_appLaunch(24,5)
    # user_appLaunch2 = get_user_from_appLaunch(25,5)
    user_appLaunch3 = get_user_from_appLaunch(26,3)
    user_appLaunch4 = get_user_from_appLaunch(27,2)
    user_appLaunch5 = get_user_from_appLaunch(28,1)
    user_appLaunch6 = get_user_from_appLaunch(29,1)
    # user_appLaunch7 = get_user_from_appLaunch(22,6)
    # user_appLaunch8 = get_user_from_appLaunch(21,7)
    # user_appLaunch9 = get_user_from_appLaunch(20,8)
    # user_appLaunch10 = get_user_from_appLaunch(19,9)
    # user_appLaunch11 = get_user_from_appLaunch(18,10)
    # user_appLaunch12 = get_user_from_appLaunch(17,11)
    # user_appLaunch13 = get_user_from_appLaunch(16,12)
    # print("app launcher {}".format(len(user_appLaunch0)))
    # print("app launcher {}".format(len(user_appLaunch1)))
    # print("app launcher {}".format(len(user_appLaunch2)))
    print("app launcher {}".format(len(user_appLaunch3)))
    print("app launcher {}".format(len(user_appLaunch4)))
    print("app launcher {}".format(len(user_appLaunch5)))
    print("app launcher {}".format(len(user_appLaunch6)))
    # print("app launcher {}".format(len(user_appLaunch7)))
    # print("app launcher {}".format(len(user_appLaunch8)))
    # print("app launcher {}".format(len(user_appLaunch9)))
    # print("app launcher {}".format(len(user_appLaunch10)))
    # print("app launcher {}".format(len(user_appLaunch11)))
    # print("app launcher {}".format(len(user_appLaunch12)))
    # print("app launcher {}".format(len(user_appLaunch13)))
    user_appLaunch = list(set(
        # user_appLaunch0+
        # user_appLaunch1+
        # user_appLaunch2+
        user_appLaunch3+
        user_appLaunch4+
        user_appLaunch5+
        user_appLaunch6
        # user_appLaunch7+
        # user_appLaunch8+
        # user_appLaunch9+
        # user_appLaunch10+
        # user_appLaunch11+
        # user_appLaunch12+
        # user_appLaunch13
    ))
    print("app launcher {}".format(len(user_appLaunch)))
    checkIn(user_videoCreate,user_appLaunch)
    # user_userRegister = get_user_from_userRegister(28)
    # user_userActivity7 = get_user_from_userActivity(22,6)
    # user_userActivity0 = get_user_from_userActivity(23,6)
    # user_userActivity1= get_user_from_userActivity(24,5)
    # user_userActivity2 = get_user_from_userActivity(25,4)
    user_userActivity3 = get_user_from_userActivity(26,2)
    user_userActivity4 = get_user_from_userActivity(27,1)
    user_userActivity5 = get_user_from_userActivity(28,1)
    user_userActivity6 = get_user_from_userActivity(29,0)
    # user_userActivity7 = get_user_from_userActivity(22,6)
    # user_userActivity8 = get_user_from_userActivity(21,7)
    # user_userActivity9 = get_user_from_userActivity(20,8)
    # user_userActivity10 = get_user_from_userActivity(19,9)
    # user_userActivity11 = get_user_from_userActivity(18,10)
    # user_userActivity12 = get_user_from_userActivity(17,11)
    # user_userActivity13 = get_user_from_userActivity(16,12)
    # print("activiter {}".format(len(user_userActivity0)))
    # print("activiter {}".format(len(user_userActivity1)))
    # print("activiter {}".format(len(user_userActivity2)))
    print("activiter {}".format(len(user_userActivity3)))
    print("activiter {}".format(len(user_userActivity4)))
    print("activiter {}".format(len(user_userActivity5)))
    print("activiter {}".format(len(user_userActivity6)))
    # print("activiter {}".format(len(user_userActivity7)))
    # print("activiter {}".format(len(user_userActivity8)))
    # print("activiter {}".format(len(user_userActivity9)))
    # print("activiter {}".format(len(user_userActivity10)))
    # print("activiter {}".format(len(user_userActivity11)))
    # print("activiter {}".format(len(user_userActivity12)))
    # print("activiter {}".format(len(user_userActivity13)))
    user_userActivity = list(set(
        # user_userActivity0+
        # user_userActivity1+
        # user_userActivity2+
        user_userActivity3+
        user_userActivity4+
        user_userActivity5+
        user_userActivity6
        # user_userActivity7+
        # user_userActivity8+
        # user_userActivity9+
        # user_userActivity10+
        # user_userActivity11+
        # user_userActivity12+
        # user_userActivity13
    ))


    print("activiter {}".format(len(user_userActivity)))
    checkIn(user_userActivity,user_appLaunch)
    users = list(set(
        user_videoCreate
        +user_appLaunch
        # +user_userRegister
        +user_userActivity))
    print("merged video creater, launcher, and activists {}".format(len(users)))


    # active_user = register_in_activity_author(23,4)
    # print("user is author {}".format(len(active_user)))

    users = list(set(
        user_videoCreate
        +user_appLaunch
        # +active_user
        # +user_userRegister
        +user_userActivity))
    print("merged video creater, launcher, activists and user&author {}".format(len(users)))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    hdf = pd.Series(users,name="user_id")
    hfile = "hCoded/hcode_v15_"+str_time + ".csv"
    hdf.to_csv(hfile,header=False,index=False)
    # merged_csv = pd.read_csv("result/submission_2018-05-31_23-40.csv",header=None,index_col=None,names=["user_id"])
    # mc = merged_csv["user_id"].tolist()
    # users = list(set(users+mc))
    # print(len(users))
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # submission_file = "result/submission_" + str_time + ".csv"
    # with open(submission_file,"a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in users:
    #         writer.writerow([i])
    # return users
# get_user()