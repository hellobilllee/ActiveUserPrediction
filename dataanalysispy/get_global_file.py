import pandas as pd
import numpy as np

user_register_log = ["user_id", "register_day", "register_type", "device_type"]
app_launch_log = ["user_id", "app_launch_day"]
video_create_log = ["user_id", "video_create_day"]
user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]


def get_global_file():
    print("get users from user register log")
    # user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8,
                           "device_type": np.uint16}
    df_user_register = pd.read_csv("data/user_register_log.csv", header=0, index_col=None, dtype=dtype_user_register)
    # df_user_register.drop_duplicates(inplace=True)
    # df_user_register_train = df_user_register.loc[(df_user_register["register_day"]>=trainSpan[0])&(df_user_register["register_day"]<=trainSpan[1])]
    # these are global features
    df_user_register["register_day_rate"] = df_user_register.groupby(by=["register_day"])["register_day"].transform(
        "count")
    df_user_register["register_type_rate"] = df_user_register.groupby(by=["register_type"])["register_type"].transform(
        "count")
    df_user_register["register_type_device"] = df_user_register.groupby(by=["register_type"])["device_type"].transform(
        lambda x: x.nunique())
    df_user_register["device_type_rate"] = df_user_register.groupby(by=["device_type"])["device_type"].transform(
        "count")
    df_user_register["device_type_register"] = df_user_register.groupby(by=["device_type"])["register_type"].transform(
        lambda x: x.nunique())
    df_user_register.to_csv("data/user_register_log_global.csv",header=True,index=False)

    user_register_feature = ["user_id",
                             "register_day_rate", "register_type_rate",
                             "register_type_device", "device_type_rate", "device_type_register"
                             ]
    df_user_register_base = df_user_register[["user_id", "register_day"]].drop_duplicates()

    print("get users from app launch log")
    # app_launch_log = ["user_id","app_launch_day"]
    dtype_app_launch = {"user_id": np.uint32, "app_launch_day": np.uint8}
    df_app_launch = pd.read_csv("data/app_launch_log.csv", header=0, index_col=None, dtype=dtype_app_launch)
    df_app_launch = df_app_launch.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)

    df_app_launch["user_app_launch_rate_global"] = df_app_launch.groupby(by=["user_id"])[
        "app_launch_day"].transform("count")
    # df_app_launch["user_app_launch_register_min_time_global"] = df_app_launch.groupby(by=["user_id"])[
    #                                                                 "app_launch_day"].transform(lambda x: min(x)) - \
    #                                                             df_app_launch["register_day"]
    df_app_launch["user_app_launch_register_max_time_global"] = df_app_launch.groupby(by=["user_id"])[
                                                                    "app_launch_day"].transform(lambda x: max(x)) - \
                                                                df_app_launch["register_day"]
    df_app_launch["user_app_launch_register_mean_time_global"] = df_app_launch.groupby(by=["user_id"])[
                                                                     "app_launch_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_app_launch["register_day"]
    df_app_launch["user_app_launch_gap_global"] = df_app_launch.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_app_launch["user_app_launch_var_global"] = df_app_launch.groupby(by=["user_id"])[
        "app_launch_day"].transform(lambda x: np.var(x))
    df_app_launch.to_csv("data/app_launch_log_global.csv", header=True, index=False)

    print("get users from video create")
    # video_create_log = ["user_id", "video_create_day"]
    dtype_video_create = {"user_id": np.uint32, "video_create_day": np.uint8}
    df_video_create = pd.read_csv("data/video_create_log.csv", header=0, index_col=None, dtype=dtype_video_create)
    df_video_create = df_video_create.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)

    df_video_create["user_video_create_rate_global"] = df_video_create.groupby(by=["user_id"])[
        "video_create_day"].transform("count")
    df_video_create["user_video_create_day_global"] = df_video_create.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: x.nunique())
    df_video_create["user_video_create_frequency_global"] = df_video_create["user_video_create_rate_global"] / \
                                                            df_video_create["user_video_create_day_global"]

    df_video_create["user_video_create_register_min_time_global"] = df_video_create.groupby(by=["user_id"])[
                                                                        "video_create_day"].transform(
        lambda x: min(x)) - \
                                                                    df_video_create["register_day"]
    df_video_create["user_video_create_register_max_time_global"] = df_video_create.groupby(by=["user_id"])[
                                                                        "video_create_day"].transform(
        lambda x: max(x)) - \
                                                                    df_video_create["register_day"]
    df_video_create["user_video_create_register_mean_time_global"] = df_video_create.groupby(by=["user_id"])[
                                                                         "video_create_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_video_create["register_day"]
    # df_video_create["user_video_create_register_mean_time"] = df_video_create["video_create_day"]-df_video_create["register_day"]
    df_video_create["user_video_create_gap_global"] = df_video_create.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_video_create["user_video_create_var_global"] = df_video_create.groupby(by=["user_id"])[
        "video_create_day"].transform(lambda x: np.var(x))
    df_video_create.to_csv("data/video_create_log_global.csv", header=True, index=False)

    print("get users from user activity log")
    # user_activity_log = ["user_id", "user_activity_day", "page", "video_id", "author_id", "action_type"]
    # usecols = ["user_id", "user_activity_day", "page","action_type"]
    dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                           "author_id": np.uint32, "action_type": np.uint8}
    df_user_activity = pd.read_csv("data/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity)
    df_user_activity = df_user_activity.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
    # df_user_activity = df_user_activity.sample(n=50000)
    print("read , merge and sample over")
    # print(df_user_activity.describe())
    # df_user_activity.drop_duplicates(inplace=True)
    # print(df_user_activity.describe())
    df_user_activity["user_activity_rate_global"] = (df_user_activity.groupby(by=["user_id"])["user_id"].transform(
        "count")).astype(np.uint16)
    df_user_activity["user_activity_day_rate_global"] = (df_user_activity.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: x.nunique())).astype(np.uint8)
    df_user_activity["user_activity_frequency_global"] = df_user_activity["user_activity_rate_global"]/df_user_activity["user_activity_day_rate_global"]
    df_user_activity["user_activity_gap_global"] = df_user_activity.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: (max(x) - min(x)) / (len(set(x)) - 1) if len(set(x)) > 1 else 0)
    df_user_activity["user_activity_var_global"] = df_user_activity.groupby(by=["user_id"])[
        "user_activity_day"].transform(lambda x: np.var(x))
    df_user_activity["user_activity_register_min_time_global"] = (df_user_activity.groupby(by=["user_id"])[
                                                                     "user_activity_day"].transform(lambda x: min(x)) - \
                                                                 df_user_activity["register_day"]).astype(np.uint8)
    df_user_activity["user_activity_register_max_time_global"] = (df_user_activity.groupby(by=["user_id"])[
                                                                     "user_activity_day"].transform(lambda x: max(x)) - \
                                                                 df_user_activity["register_day"]).astype(np.uint8)
    df_user_activity["user_activity_register_mean_time_global"] = df_user_activity.groupby(by=["user_id"])[
                                                                      "user_activity_day"].transform(
        lambda x: (max(x) + min(x)) / 2) - df_user_activity["register_day"]
    print("groupby one columns ")
    df_user_activity["user_page_num_global"] = (df_user_activity.groupby(by=["user_id"])["page"].transform(
        lambda x: x.nunique())).astype(np.uint8)
    df_user_activity["user_video_num_global"] = (df_user_activity.groupby(by=["user_id"])["video_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity["user_author_num_global"] = (df_user_activity.groupby(by=["user_id"])["author_id"].transform(
        lambda x: x.nunique())).astype(np.uint16)
    df_user_activity["user_action_type_num_global"] = (df_user_activity.groupby(by=["user_id"])[
        "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    print("groupby two columns ")
    # df_user_activity["user_author_video_num_global"] = (df_user_activity.groupby(by=["user_id", "author_id"])[
    #     "video_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint16)
    # print("1")
    # df_user_activity["user_video_action_type_num_global"] = (df_user_activity.groupby(by=["user_id", "video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # print("2")
    # df_user_activity["user_author_action_type_num_global"] = (df_user_activity.groupby(by=["user_id", "author_id"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # print("3")
    # df_user_activity["user_page_action_type_num_global"] = (df_user_activity.groupby(by=["user_id", "page"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    print("data process over")
    # df_user_activity["page_rate_global"] = (df_user_activity.groupby(by=["page"])["page"].transform("count")).astype(np.uint32)
    # df_user_activity["page_video_global"] = (df_user_activity.groupby(by=["page"])["video_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint32)
    # df_user_activity["page_author_global"] = (df_user_activity.groupby(by=["page"])["author_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint32)
    # df_user_activity["video_rate_global"] = (df_user_activity.groupby(by=["video_id"])["video_id"].transform(
    #     "count")).astype(np.uint32)
    # df_user_activity["video_user_global"] = (df_user_activity.groupby(by=["video_id"])["user_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint16)
    # df_user_activity["video_action_type_global"] = (df_user_activity.groupby(by=["video_id"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity["author_rate_global"] = (df_user_activity.groupby(by=["video_id"])["author_id"].transform(
    #     "count")).astype(np.uint32)
    # df_user_activity["author_user_global"] = (df_user_activity.groupby(by=["author_id"])["user_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint16)
    # df_user_activity["author_video_global"] = (df_user_activity.groupby(by=["author_id"])["video_id"].transform(
    #     lambda x: x.nunique())).astype(np.uint16)
    # df_user_activity["author_action_type_global"] = (df_user_activity.groupby(by=["author_id"])[
    #     "action_type"].transform(lambda x: x.nunique())).astype(np.uint8)
    # df_user_activity["action_type_rate_global"] = (df_user_activity.groupby(by=["action_type"])[
    #     "action_type"].transform("count")).astype(np.uint32)
    df_user_activity.to_csv("data/user_activity_log_global.csv", header=True, index=False)

if __name__ == "__main__":
    get_global_file()
