import pandas as pd

# train_set = pd.read_csv("data/training_m1-23.csv", header=0, index_col=None)
# # del train_set1,train_set2
# # gc.collect()
# print(train_set.describe())
# keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id"]))
# print("begin to drop the duplicates")
# train_set.drop_duplicates(subset=keep_feature, inplace=True)
# print(train_set.describe())
# train_label = train_set["label"]
# train_set = train_set.drop(labels=["label", "user_id"], axis=1)
#
# ls = [0,1,2,3,4,5,1,2,5,1,2,4,9]
# print(ls.count(10)/len(ls))
import numpy as np
print("get users from user activity log")
dtype_user_activity = {"user_id": np.uint32, "user_activity_day": np.uint8, "page": np.uint8, "video_id": np.uint32,
                       "author_id": np.uint32, "action_type": np.uint8}
df_user_activity = pd.read_csv("data/user_activity_log.csv", header=0, index_col=None, dtype=dtype_user_activity)
# df_user_activity = df_user_activity.merge(df_user_register_base, on=["user_id"], how="left").fillna(-1)
df_user_activity_train = df_user_activity.loc[
    (df_user_activity["user_activity_day"] >= 1) & (
            df_user_activity["user_activity_day"] <= 9)]
print(df_user_activity_train.describe())
user_activity_author = df_user_activity_train["author_id"].unique().tolist()
print(user_activity_author)
df_user_activity_train["user_in_author"] = 0
# df_user_activity_train["user_in_author"] = df_user_activity_train["user_id"].apply(lambda x: 1 if x in user_activity_author else 0)
print("begin to get user in author or not mark")
df_user_activity_train.loc[df_user_activity_train["user_id"].isin(user_activity_author),"user_in_author"]=1
print(df_user_activity_train.describe())