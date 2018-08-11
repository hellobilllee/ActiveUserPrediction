import csv
import datetime
# import gc
import pandas as pd
import joblib
import lightgbm
from lightgbm import LGBMClassifier,cv
from scipy.stats import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from skopt import BayesSearchCV
from skopt.callbacks import  DeltaXStopper
from data_process_v2 import processing
from skopt.space import Categorical

def predict(clf2, test_set):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    print("begin to make predictions")
    res = clf2.predict(test_set.values)
    uid["y_hat"] = pd.Series(res)
    uid["label"] = uid.groupby(by=["user_id"])["y_hat"].transform(lambda x: stats.mode(x)[0][0])
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    uid_file = "result/uid_" + str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    active_users = (uid.loc[uid["label"] == 1]).user_id.unique().tolist()
    print(len(active_users))
    print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    submission_file = "result/submission_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])
# using this module ,one needs to deconstruct some of the features in data_process
keep_feature = ["user_id",
                "register_day_rate", "register_type_rate",
                "register_type_device", "device_type_rate", "device_type_register",
                "user_app_launch_register_mean_time",
                "user_app_launch_rate", "user_app_launch_gap",
                "user_video_create_register_mean_time",
                "user_video_create_rate", "user_video_create_day", "user_video_create_gap",
                 "user_activity_register_mean_time", "user_activity_rate",
                 "user_activity_frequency",
                 "user_activity_day_rate", "user_activity_gap",
                 "user_page_num", "user_video_id_num",
                 "user_author_id_num", "user_author_id_video_num",
                 "user_action_type_num"
                  ]
def run():
    print("begin to load the trainset1")
    train_set1 = processing(trainSpan=(1,12),label=True)
    # print(train_set1.describe())
    print("begin to load the trainset2")
    train_set2 = processing(trainSpan=(13,23),label=True)
    # print(train_set2.describe())
    print("begin to merge the trainsets")
    train_set = pd.concat([train_set1,train_set2],axis=0)
    print(train_set.describe())
    # del train_set1,train_set2
    # gc.collect()
    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    train_label =train_set["label"]
    train_set = train_set.drop(labels=["label","user_id"], axis=1)

    # train_x, val_x,train_y,val_y = train_test_split(train_set.values,train_label.values,test_size=0.33,random_state=42,shuffle=True)
    print("begin to make prediction with plain features and without tuning parameters")
    initial_params = {
        "n_jobs": -1,
        "n_estimators": 400,
        "criterion": "gini",
        "max_features": 'auto',
        "max_depth": 6,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": 64,
        "min_impurity_decrease": 0.0,
    }
    # train_data = lightgbm.Dataset(train_set.values, label=train_label.values, feature_name=list(train_set.columns))

    scoring = {'AUC': 'roc_auc', 'f1': "f1"}
    clf1 = GridSearchCV(RandomForestClassifier(**initial_params),
                      param_grid={"n_estimators":[400,600],"max_leaf_nodes": [16,24,32,64]},
                      scoring=scoring, cv=3, refit='f1',n_jobs=-1,verbose=0)
    clf1.fit(train_set.values, train_label.values)
    # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
    # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
    print(clf1.best_score_)
    print(clf1.best_params_)
    # clf1 = LGBMClassifier(**initial_params)
    # clf1.fit(X=train_x,y=train_y,eval_set=(val_x,val_y),early_stopping_rounds=20,eval_metric="auc")
    print("load the test dataset")
    test_set = processing(trainSpan=(20, 30), label=False)
    print("begin to make prediction")
    predict(clf1,test_set)

    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    feature_importances = clf1.best_estimator_.feature_importances_
    print(feature_importances)
    print(feature_names)

    with open("kuaishou_stats.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["feature importance of catboost for tencent-crt", str_time])
        # writer.writerow(eval_metrics)
        feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
        for score, name in feature_score_name:
            print('{}: {}'.format(name, score))
            writer.writerow([name, score])
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)

    print("begin to tune the parameters with the selected feature")
    paramsSpace = {
        "n_estimators": (200, 800),
        "criterion": Categorical(["gini", "entropy"]),
        "max_features": (0.6, 1.0, 'uniform'),
        "max_depth": (3, 8),
        "min_samples_split": (2, 128),
        "min_samples_leaf": (1, 128),
        "min_weight_fraction_leaf": (0.0, 0.5, 'uniform'),
        "max_leaf_nodes": (16, 128),
        "min_impurity_decrease": (1e-6, 1e-1, 'log-uniform'),
    }

    def tune_parameter(X, y, clf, params):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        gs = BayesSearchCV(
            estimator=clf, search_spaces=params,
            scoring="f1", n_iter=60,optimizer_kwargs={"base_estimator":"RF"},
            verbose=0, n_jobs=-1, cv=3, refit=True, random_state=1234
        )
        gs.fit(X, y,callback=DeltaXStopper(0.000001))
        best_params = gs.best_params_
        best_score = gs.best_score_
        print(best_params)
        print(best_score)
        str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        with open("kuaishou_stats.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["the best params for lightgbm: "])
            for key, value in best_params.items():
                writer.writerow([key, value])
            writer.writerow(["the best score for lightgbm: ", best_score,str_time])
        return gs

    model = RandomForestClassifier(**initial_params)
    clf2 = tune_parameter(train_set.values,train_label.values,model,paramsSpace)
    print("parameter tuning over, begin to save the model!")
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

    model_name = "lightgbm_" + str_time + ".pkl"
    joblib.dump(clf2, model_name)

    print("begin to process the whole dataset and ready to feed into the fitted model")
    predict(clf2,test_set)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print("begin to get important features")
    feature_names = train_set.columns
    feature_importances = clf2.best_estimator_.feature_importances_
    print(feature_importances)
    print(feature_names)

    with open("kuaishou_stats.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["feature importance of catboost for tencent-crt", str_time])
        # writer.writerow(eval_metrics)
        feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
        for score, name in feature_score_name:
            print('{}: {}'.format(name, score))
            writer.writerow([name, score])
    sorted_feature_name = [name for score, name in feature_score_name]
    print(sorted_feature_name)
if __name__=="__main__":
    run()