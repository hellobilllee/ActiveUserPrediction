import csv
import datetime
import pandas as pd
import joblib
from scipy.stats import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import  DeltaXStopper
from data_process_v7 import processing
from skopt.space import Categorical, Real
import numpy as np

def predict(clf2, test_set):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    print("begin to make predictions")
    res = clf2.predict_proba(test_set.values)
    uid["proba1"] = pd.Series(res[:, 1])
    uid["score"] = uid.groupby(by=["user_id"])["proba1"].transform(lambda x: sum(x) / float(len(x)))
    uid.drop_duplicates(subset=["user_id"],inplace=True)
    uid.sort_values(by=["score"],axis=0,ascending=False,inplace=True)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    uid_file = "result/uid_" + str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    active_users = uid.loc[uid["score"]>0.5]["user_id"].unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission_file = "result/submission_lgb_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])

def run():
    use_feature = [
        "user_id","label",
        "register_day_type_rate",
        "register_day_type_ratio",
        # "register_day_device_rate",
        "register_day_device_ratio",
        "register_type_ratio",
        "register_type_device",
        "register_type_device_ratio",
        # "device_type_ratio",
        # "device_type_register",
        "device_type_register_ratio",
        # "register_day_register_type_device_rate",
        "register_day_register_type_device_ratio",
        # "register_day_device_type_register_rate",
        # "register_day_device_type_register_ratio",

        "user_app_launch_rate",
        "user_app_launch_ratio",
        "user_app_launch_gap",
        "user_app_launch_var",

        "user_app_launch_count_b1",
        "user_app_launch_count_b2",
        "user_app_launch_count_b3",
        "user_app_launch_count_b4",
        "user_app_launch_count_b5",
        "user_app_launch_count_b6",
        "user_app_launch_count_b7",

        "user_app_launch_count_f1",
        "user_app_launch_count_f2",
        "user_app_launch_count_f3",
        "user_app_launch_count_f4",
        "user_app_launch_count_f5",
        "user_app_launch_count_f6",
        "user_app_launch_count_f7",

        "user_video_create_rate",
        "user_video_create_ratio",
        "user_video_create_day",
        "user_video_create_day_ratio",
        "user_video_create_frequency",
        "user_video_create_gap",
        "user_video_create_day_var",
        "user_video_create_var",

        "user_video_create_count_b1",
        "user_video_create_count_b2",
        "user_video_create_count_b3",
        "user_video_create_count_b4",
        "user_video_create_count_b5",
        "user_video_create_count_b6",
        "user_video_create_count_b7",

        "user_video_create_count_f1",
        "user_video_create_count_f2",
        "user_video_create_count_f3",
        "user_video_create_count_f4",
        "user_video_create_count_f5",
        # "user_video_create_count_f6",
        # "user_video_create_count_f7",

        "user_activity_rate",
        "user_activity_ratio",
        "user_activity_var",
        "user_activity_day_rate",
        "user_activity_day_ratio",
        "user_activity_frequency",
        "user_activity_gap",
        "user_activity_day_var",
        "user_page_num",
        "user_page_day_ratio",
        "user_video_num",
        "user_video_num_ratio",
        "user_author_num",
        "user_author_num_ratio",
        "user_action_type_num",

        "user_activity_count_b1",
        "user_activity_count_b2",
        "user_activity_count_b3",
        "user_activity_count_b4",
        "user_activity_count_b5",
        "user_activity_count_b6",
        "user_activity_count_b7",

        "user_activity_count_f1",
        "user_activity_count_f2",
        "user_activity_count_f3",
        "user_activity_count_f4",
        "user_activity_count_f5",
        "user_activity_count_f6",
        "user_activity_count_f7"
    ]
    print("begin to load the trainset1")
    # train_set1 = processing(trainSpan=(1,19),label=True)
    # train_set1.to_csv("data/training_eld1-19_r.csv", header=True, index=False)
    # train_set1 = pd.read_csv("data/training_eld1-22.csv", header=0, index_col=None, usecols=use_feature)
    train_set1 = pd.read_csv("../data/data_v3/training_eld1-15_r.csv", header=0, index_col=None)
    # print(train_set1.columns)
    # print(True if "label" in train_set1.columns else False)
    print(train_set1.describe())
    print("begin to load the trainset2")
    # train_set2 = processing(trainSpan=(1,20),label=True)
    # train_set2.to_csv("data/training_eld1-20_r.csv", header=True, index=False)
    # train_set2 = pd.read_csv("data/training_rld1-20.csv", header=0, index_col=None, usecols=use_feature)
    train_set2 = pd.read_csv("../data/data_v3/training_eld1-16_r.csv", header=0, index_col=None)
    print(train_set2.describe())
    print("begin to load the trainset3")
    # train_set3 = processing(trainSpan=(1,21),label=True)
    # train_set3.to_csv("data/training_eld1-21_r.csv", header=True, index=False)
    # train_set3 = pd.read_csv("data/training_ld1-17.csv", header=0, index_col=None, usecols=use_feature)
    train_set3 = pd.read_csv("../data/data_v3/training_eld1-18_r.csv", header=0, index_col=None)
    print(train_set3.describe())
    print("begin to load the trainset4")
    # train_set4 = processing(trainSpan=(1,22),label=True)
    # train_set4.to_csv("data/training_eld1-22_r.csv", header=True, index=False)
    # train_set4 = pd.read_csv("data/training_rld1-19.csv", header=0, index_col=None, usecols=use_feature)
    train_set4 = pd.read_csv("../data/data_v3/training_eld1-19_r.csv", header=0, index_col=None)
    print(train_set4.describe())
    # print("begin to load the trainset5")
    # train_set5 = processing(trainSpan=(1,21),label=True)
    # train_set5.to_csv("data/training_ld1-21.csv", header=True, index=False)
    # train_set5 = pd.read_csv("data/training_eld1-22.csv", header=0, index_col=None, usecols=use_feature)
    # train_set5 = pd.read_csv("data/training_eld1-22_r.csv", header=0, index_col=None)
    # print(train_set5.describe())
    print("begin to load the validation set")
    # train_set_feature = train_set5.columns
    # val_set = processing(trainSpan=(1,23),label=True)
    # val_set.to_csv("data/training_eld1-23.csv", header=True, index=False)
    # val_set = pd.read_csv("data/training_eld1-23.csv", header=0, index_col=None, usecols=use_feature)
    # val_set = pd.read_csv("data/training_eld1-23_r.csv", header=0, index_col=None,usecols=train_set_feature)
    val_set = pd.read_csv("../data/data_v3/training_eld1-23_r.csv", header=0, index_col=None)
    print(val_set.describe())
    train_set = pd.concat([train_set1, train_set2, train_set3, train_set4], axis=0)
    # train_set = pd.concat([ train_set2, train_set3, train_set4, train_set5], axis=0)
    # train_set = pd.concat([ train_set5], axis=0)
    ds = train_set.describe()
    print(ds)
    # train_set = pd.concat([train_set1], axis=0)
    # train_set = pd.concat([train_set1,train_set2],axis=0)
    # train_set.to_csv("data/training_lm1-11_12-23.csv", header=True, index=False)
    # train_set = pd.read_csv("data/training_m1-23.csv", header=0, index_col=None)
    # del train_set1,train_set2
    # gc.collect()
    print(train_set.describe())
    keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id", "label"]))

    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature, inplace=True)
    val_set.drop_duplicates(subset=keep_feature, inplace=True)
    print(train_set.describe())
    print(val_set.describe())
    train_label = train_set["label"]
    val_label = val_set["label"]
    train_set = train_set.drop(labels=["label", "user_id"], axis=1)
    val_set = val_set.drop(labels=["label", "user_id"], axis=1)

    # keep_feature = []
    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.25,random_state=42,shuffle=False)
    # train_set = train_set[keep_feature]
    # feature_names = train_set.columns

    for fea in keep_feature:
        train_set[fea] = (train_set[fea] - train_set[fea].min()) / (train_set[fea].max() - train_set[fea].min())
        val_set[fea] = (val_set[fea] - val_set[fea].min()) / (val_set[fea].max() - val_set[fea].min())
    # sel = VarianceThreshold(threshold=0.00001)
    # train_set_cols = train_set.columns
    # val_set_cols = train_set.columns
    # train_set = sel.fit_transform(train_set.values)
    # val_set = sel.transform(val_set.values)
    # train_x, val_set,train_label,val_label = train_test_split(train_set.values,train_label.values,test_size=0.25,random_state=42,shuffle=False)
    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.33,random_state=42,shuffle=True)
    print("begin to make prediction with plain features and without tuning parameters")
    initial_params = {
        "n_jobs": -1,
        "max_iter": 1200,
        "solver": 'liblinear',
        "C": 0.01,
        # "class_weight":{1:0.2,0:1},
    }
    # train_data = lightgbm.Dataset(train_set.values, label=train_label.values, feature_name=list(train_set.columns))

    # scoring = {'f1': "f1"}
    # clf1 = GridSearchCV(LogisticRegression(**initial_params),
    #                   param_grid={
    #                       "max_iter":[1200],
    #                       # "solver": ['liblinear'],
    #                       "solver": ['liblinear'],
    #                       # "C":[0.012,0.01,0.008,0.014,0.006],
    #                       # "C":[1.4,1.5,1.6,1.7,1.8,2.0],
    #                       # "C":[1.0,1.2,1.5,1.8,2,2.2,2.5,2.8,3,3.5,4,6,8],
    #                       "C":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    #                       # "C":[0.007,0.008,0.009],
    #                       # "penalty":["l1"],
    #                       "penalty":["l2"]
    #                       # "class_weight": [{1: 1.3, 0: 1},{1: 1.4, 0: 1},{1: 1.25, 0: 1},{1: 1.35, 0: 1},{1: 1.2, 0: 1}]
    #                   },
    #                   scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=2)

    clf1 = LogisticRegression(**initial_params)
    clf1.fit(train_set.values, train_label.values)
    # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
    # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
    # bs = clf1.best_score_
    # print(bs)
    # bp = clf1.best_params_
    # print(bp)
    # clf1 = LGBMClassifier(**initial_params)
    # clf1.fit(X=train_x,y=train_y,eval_set=(val_x,val_y),early_stopping_rounds=20,eval_metric="auc")
    clf1.fit(train_set.values, train_label.values)
    print("train set report:")
    yhat = clf1.predict(train_set.values)
    print(classification_report(y_pred=yhat, y_true=train_label.values,digits=4))
    print("validation set report:")
    yhat = clf1.predict(val_set.values)
    # yhat = clf1.predict(val_set)
    print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))


    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    # feature_importances = np.reshape(clf1.best_estimator_.coef_, -1)
    feature_importances = np.reshape(clf1.coef_, -1)
    # print(feature_importances)
    # print(feature_names)
    feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    for score, name in feature_score_name:
        print('{}: {}'.format(name, score))
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)

    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of lr for kuaishou ", str_time])
    #     writer.writerow(["best score", bs, "best params"])
    #     for key, value in bp.items():
    #         writer.writerow([key, value])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)
    # clf1 = LogisticRegression(**bp)
    # train_set["label"] = train_label
    # val_set["label"] = val_label
    # train_set = pd.concat([train_set, val_set], axis=0)
    # train_label = train_set["label"]
    # train_set = train_set.drop(labels=["label"], axis=1)
    # clf1.fit(train_set.values, train_label.values)
    # # test_set = processing(trainSpan=(1, 30), label=False)
    # # test_set.to_csv("data/testing_ld1-30.csv",header=True,index=False)
    # # test_set = pd.read_csv("data/testing_w11.csv",header=0,index_col=None)
    # test_set = pd.read_csv("data/testing_ld1-30.csv",header=0,index_col=None,usecols=keep_feature+["user_id"])
    # # # test_set.drop_duplicates(inplace=True)
    # for fea in keep_feature:
    #     test_set[fea] = (test_set[fea]-test_set[fea].min())/(test_set[fea].max()-test_set[fea].min())
    # print("begin to make prediction")
    # predict(clf1,test_set)


    #
    # print("begin to tune the parameters")
    # paramsSpace = {
    #             "C":  (1.0, 1e+1),
    #             # "max_iter":(200, 800),
    #             # "solver": Categorical(["liblinear"]),
    #         }
    #
    # def tune_parameter(X, y, clf, params):
    #     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #     gs = BayesSearchCV(
    #         estimator=clf, search_spaces=params,
    #         scoring="f1", n_iter=100,optimizer_kwargs={"base_estimator":"GP"},
    #         verbose=2, n_jobs=-1, cv=3, refit=True, random_state=1234
    #     )
    #     gs.fit(X, y,callback=DeltaXStopper(0.000001))
    #     best_params = gs.best_params_
    #     best_score = gs.best_score_
    #     print(best_params)
    #     print(best_score)
    #     str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    #     with open("kuaishou_stats.csv", 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["the best params for lightgbm: "])
    #         for key, value in best_params.items():
    #             writer.writerow([key, value])
    #         writer.writerow(["the best score for lightgbm: ", best_score,str_time])
    #     return gs
    #
    # model = LogisticRegression(**bp)
    # clf2 = tune_parameter(train_set.values,train_label.values,model,paramsSpace)
    # print("parameter tuning over, begin to save the model!")
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    #
    # model_name = "lr_" + str_time + ".pkl"
    # joblib.dump(clf2, model_name)
    #
    # print("begin to process the whole dataset and ready to feed into the fitted model")
    # predict(clf2,test_set)
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # print("begin to get important features")
    # feature_names = train_set.columns
    # feature_importances = np.reshape(clf2.best_estimator_.coef_,-1)
    # print(feature_importances)
    # print(feature_names)
    #
    # with open("kuaishou_stats.csv", 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["feature importance of lr for kuaishou", str_time])
    #     # writer.writerow(eval_metrics)
    #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    #     for score, name in feature_score_name:
    #         print('{}: {}'.format(name, score))
    #         writer.writerow([name, score])
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)
if __name__=="__main__":
    run()