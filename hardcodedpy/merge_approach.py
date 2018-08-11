import csv
import datetime
from hardcode_approach import get_user
import pandas as pd
import numpy as np
def merge1():
    hardcode_user = get_user()
    print(len(hardcode_user))
    merged_csv1 = pd.read_csv("result/submission_2018-06-01_17-07.csv",header=None,index_col=None,names=["user_id"])
    mc1 = merged_csv1["user_id"].tolist()
    print(len(mc1))
    merged_csv2 = pd.read_csv("result/submission_2018-06-01_17-47.csv",header=None,index_col=None,names=["user_id"])
    mc2 = merged_csv2["user_id"].tolist()
    print(len(mc2))
    mc2 = [e for e in mc2 if e in mc1]
    print(len(mc2))
    merged_csv3 = pd.read_csv("result/submission_2018-06-01_18-05catboost.csv",header=None,index_col=None,names=["user_id"])
    mc3 = merged_csv3["user_id"].tolist()
    print(len(mc3))
    mc3 = [e for e in mc3 if e in mc2]
    print(len(mc3))
    users = list(set(hardcode_user+mc3))
    print(len(users))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_" + str_time + ".csv"
    with open(submission_file,"a",newline="") as f:
        writer = csv.writer(f)
        for i in users:
            writer.writerow([i])
def merge2():
    hardcode_user = get_user()
    print(len(hardcode_user))
    merged_csv1 = pd.read_csv("result/submission_2018-05-30_23-20.csv",header=None,index_col=None,names=["user_id"])
    mc1 = merged_csv1["user_id"].tolist()
    print(len(mc1))
    merged_csv2 = pd.read_csv("merge/submission_2018-06-01_11-57.csv",header=None,index_col=None,names=["user_id"])
    mc2 = merged_csv2["user_id"].tolist()
    print(len(mc2))
    mc2 = [e for e in mc2 if e in mc1]
    print(len(mc2))
    users = list(set(hardcode_user+mc2))
    print(len(users))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_" + str_time + ".csv"
    with open(submission_file,"a",newline="") as f:
        writer = csv.writer(f)
        for i in users:
            writer.writerow([i])
def merge3():
    hardcode_user = get_user()
    print(len(hardcode_user))
    merged_csv1 = pd.read_csv("result/submission_lgb_2018-06-03_00-34.csv",header=None,index_col=None,names=["user_id"])
    mc1 = merged_csv1["user_id"][:23500].tolist()
    print(len(mc1))
    users = list(set(hardcode_user+mc1))
    print(len(users))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_" + str_time + ".csv"
    # with open(submission_file,"a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in users:
    #         writer.writerow([i])
def merge4():
    hardcode_user = get_user()
    print(len(hardcode_user))
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    hdf = pd.Series(hardcode_user,name="user_id")
    hfile = "hCoded/hcode_"+str_time + ".csv"
    hdf.to_csv(hfile,header=True,index=False)
    merged_csv1 = pd.read_csv("lgb/uid_2018-06-11_22-04-13.csv",header=0,index_col=None)
    mc1 = merged_csv1["user_id"][:23800].tolist()
    print(len(mc1))
    users = list(set(hardcode_user+mc1))
    print(len(users))

    submission_file = "merge/submission_" + str_time + ".csv"
    # with open(submission_file,"a",newline="") as f:
    #     writer = csv.writer(f)
    #     for i in users:
    #         writer.writerow([i])
# merge4()
def merge5():
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # hardcode_user = get_user()
    # hardcode_user = pd.read_csv("merge/submission_v5_fre2_2018-06-08_11-48.csv",header=0,index_col=None)["user_id"].tolist()
    # print(len(hardcode_user))
    mc2 = pd.read_csv("hCoded/hcode_v12_lastdayofactivityandlaunchcount1_withauthor_2018-06-16_08-54.csv",header=None,index_col=None,names=["user_id"])["user_id"].unique().tolist()
    print(len(mc2))
    # hdf = pd.Series(hardcode_user,name="user_id")
    # hfile = "hCoded/hcode_28ac_"+str_time + ".csv"
    # hdf.to_csv(hfile,header=True,index=False)
    mc = pd.read_csv("single/submission_18-23slgb_0.81-2018-06-16_08-16.csv",header=None,index_col=None,names=["user_id"])["user_id"].tolist()[:20000]
    # mc1 = pd.read_csv("lgb/uid_2018-06-04_16-55-34.csv",header=0,index_col=None)
    # mc = mc1.loc[mc1["score"]>0.40]["user_id"].tolist()
    print(len(mc))
    users = list(set(mc2+mc))
    print(len(users))
    #
    submission_file = "merge/submission_0.81lgb20000_v12_" + str_time + ".csv"

    with open(submission_file,"a",newline="") as f:
        writer = csv.writer(f)
        for i in users:
            writer.writerow([i])
# merge5()
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
def single():
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    single_csv1 = pd.read_csv("06-17/uid_2018-06-17_01-01-33.csv",header=0,index_col=None)["user_id"].unique().tolist()
    # mc1 = single_csv1.loc[single_csv1["score"]>0.48]["user_id"].tolist()
    mc1 = single_csv1[:23727]
    print(len(mc1))
    # user_userActivity = register_in_activity_author(23,2)

    users = list(set(mc1))
    print(len(users))
    submission_file = "single/submission_slgb_all0.8-" + str_time + ".csv"
    with open(submission_file,"a",newline="") as f:
        writer = csv.writer(f)
        for i in users:
            writer.writerow([i])
single()
