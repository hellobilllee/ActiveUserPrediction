import csv
import datetime
import pandas as pd

def merge5():
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # hardcode_user = get_user()
    hardcode_user = pd.read_csv("hCoded/hcode_20-29_v5_2018-06-06_20-12_nolastdayoflaunch_22-30.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()
    print(len(hardcode_user))
    # hdf = pd.Series(hardcode_user,name="user_id")
    # hfile = "hCoded/hcode_28ac_"+str_time + ".csv"
    # hdf.to_csv(hfile,header=True,index=False)
    # mc1 = pd.read_csv("lr/uid_2018-06-07_22-55-45.csv",header=0,index_col=None)
    mc = pd.read_csv("hCoded/submission_freqUsers_v2_2018-06-08_11-38.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()
    # mc = mc1.loc[mc1["score"]>0.20]["user_id"].tolist()
    print(len(mc))
    ac_users = list(set(hardcode_user)-set(mc))
    print(len(ac_users))

    v5_user = pd.read_csv("hCoded/hcode_20-29_v5_2018-06-06_20-12_nolastdayoflaunch_22-30.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()
    print(len(v5_user))

    users = list(set(v5_user+ac_users))
    print(len(users))
    # #
    # submission_file = "merge/submission_0.815-baseline+v5_" + str_time + ".csv"
    # with open(submission_file,"a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in users:
    #         writer.writerow([i])
def merge6():
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # hardcode_user = get_user()
    hardcode_user = pd.read_csv("merge/submission_rule_consec_2018-06-25_20-23.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()
    mc = pd.read_csv("lgb/submission_lgb_r3_1600_4_2018-06-24_23-42-42.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()[:24000]
    print(len(hardcode_user))
    # hdf = pd.Series(hardcode_user,name="user_id")
    # hfile = "hCoded/hcode_28ac_"+str_time + ".csv"
    # hdf.to_csv(hfile,header=True,index=False)
    # mc1 = pd.read_csv("lr/uid_2018-06-07_22-55-45.csv",header=0,index_col=None)
    # mc = pd.read_csv("hCoded/submission_freqUsers_v3_2018-06-08_11-41.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()
    # mc = mc1.loc[mc1["score"]>0.20]["user_id"].tolist()

    # mc1 = pd.read_csv("lgb/uid_2018-06-04_16-55-34.csv",header=0,index_col=None)
    # mc = mc1.loc[mc1["score"]>0.7]["user_id"].tolist()
    # merged_csv1 = pd.read_csv("lgb/uid_2018-06-11_22-04-13.csv",header=0,index_col=None)
    # mc = merged_csv1["user_id"][:23800].tolist()
    print(len(mc))

    users = list(set(hardcode_user+mc))
    # users = list(set(mc))
    print(len(users))
    # #
    submission_file = "merge/submission_lgbhest_ru_" + str_time + ".csv"
    with open(submission_file,"a",newline="") as f:
        writer = csv.writer(f)
        for i in users:
            writer.writerow([i])
# merge6()
import numpy as np
def get_user_from_activity_new(trainSpan,laterThanDay,activityCount):
    print("get users from user activity log")
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id": np.uint32, "action_type": np.uint8}
    use_feature = ["user_id","user_activity_day"]
    df_user_activity = pd.read_csv("data/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity,usecols=use_feature)

    df_user_activity = df_user_activity.loc[
        (df_user_activity["user_activity_day"] >= trainSpan[0]) & (
        df_user_activity["user_activity_day"] <= trainSpan[1])]
    # print(df_app_launch.groupby(by=["user_id"]).size())
    # print(df_app_launch.groupby(by=["app_launch_day"]).size())
    df_user_activity["activityCount"] = df_user_activity.groupby(by=["user_id"])["user_activity_day"].transform("count")
    # print(df_user_activity.describe())
    user_activity = (df_user_activity.loc[(df_user_activity["user_activity_day"]>laterThanDay)&(df_user_activity["activityCount"]>activityCount)]).user_id.unique().tolist()
    print("users active no later than {} and active for more than {} times: {} ".format(laterThanDay,activityCount,len(user_activity)))
    return user_activity
def get_user_from_appLaunch_new(trainSpan,laterThanDay,launchCount):
    print("get users from app launch log")
    app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch =  {"user_id":np.uint32,"app_launch_day":np.uint8}
    df_app_launch = pd.read_table("data/app_launch_log.txt",header=None,names=app_launch_log,index_col=None,dtype=dtype_app_launch).drop_duplicates()
    df_app_launch = df_app_launch.loc[
        (df_app_launch["app_launch_day"] >= trainSpan[0]) & (df_app_launch["app_launch_day"] <= trainSpan[1])]
    # print(df_app_launch.groupby(by=["user_id"]).size())
    # print(df_app_launch.groupby(by=["app_launch_day"]).size())
    df_app_launch["launchCount"] = df_app_launch.groupby(by=["user_id"])["app_launch_day"].transform(lambda x: x.nunique())
    user_appLaunch = (df_app_launch.loc[(df_app_launch["app_launch_day"]>laterThanDay)&(df_app_launch["launchCount"]>launchCount)]).user_id.unique().tolist()
    print("users launched no later than {} and launched for more than {} days: {} ".format(laterThanDay,launchCount,len(user_appLaunch)))
    return user_appLaunch
if __name__=="__main__":
    # av1 = get_user_from_activity_new((30,30),29,216)
    # av2 = get_user_from_activity_new((29,30),29,342)
    # av3 = get_user_from_activity_new((28,30),28,452)
    # av4 = get_user_from_activity_new((27,30),27,569)
    # av = list(set(av1+av2+av3+av4))
    # print(len(av))
    # la1 = get_user_from_appLaunch_new((29,30), 29, 1)
    # print("number of users between {} and {} is {}".format(29,30,len(la1)))
    # la2 = get_user_from_appLaunch_new((28,30), 28, 1)
    # print("number of users between {} and {} is {}".format(28,30,len(la2)))
    # la3 = get_user_from_appLaunch_new((27,30), 27, 2)
    # print("number of users between {} and {} is {}".format(27,30,len(la3)))
    # la4 = get_user_from_appLaunch_new((26,30), 26, 3)
    # print("number of users between {} and {} is {}".format(26,30,len(la4)))
    # la5 = get_user_from_appLaunch_new((25,30), 25, 4)
    # print("number of users between {} and {} is {}".format(25,30,len(la5)))
    # # # la6 = get_user_from_appLaunch_new((24,30), 24, 5)
    # # # print("number of users between {} and {} is {}".format(24,30,len(la6)))
    # # # la7 = get_user_from_appLaunch_new((23,30), 23, 6)
    # # # print("number of users between {} and {} is {}".format(23,30,len(la7)))
    # # # la8 = get_user_from_appLaunch_new((22,30), 22, 7)
    # # # print("number of users between {} and {} is {}".format(22,30,len(la8)))
    # # # la9 = get_user_from_appLaunch_new((21,30), 21, 8)
    # # # print("number of users between {} and {} is {}".format(21,30,len(la9)))
    # # # la10 = get_user_from_appLaunch_new((20,30), 20, 9)
    # # # print("number of users between {} and {} i0s {}".format(20,30,len(la10)))
    # # # la11 = get_user_from_appLaunch_new((19,30), 19, 10)
    # # # print("number of users between {} and {} is {}".format(19,30,len(la11)))
    # # # la12 = get_user_from_appLaunch_new((18,30), 18, 11)
    # # # print("number of users between {} and {} is {}".format(18,30,len(la12)))
    # # # la13 = get_user_from_appLaunch_new((17,30), 17, 12)
    # # # print("number of users between {} and {} is {}".format(17,30,len(la13)))
    # # # la = list(set(av1+av2+la1+la2+la3+la4+la5+la6+la7+la8+la9+la10+la11+la12+la13))
    # la = list(set(av+la1+la2+la3+la4+la5))
    # print("number of consecutive users {}".format(len(la)))
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # submission_file = "merge/submission_rule_consec_" + str_time + ".csv"
    # with open(submission_file,"a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in la:
    #         writer.writerow([i])
    merge6()