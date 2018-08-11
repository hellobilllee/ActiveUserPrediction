import csv
import datetime
import pandas as pd
# import joblib
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report, f1_score
# from sklearn.model_selection import GridSearchCV, train_test_split
# from skopt import BayesSearchCV
# from skopt.callbacks import  DeltaXStopper
# from data_process_v7 import processing
# from sklearn.feature_selection import VarianceThreshold
import numpy as np
# def predict(clf2, test_set,param):
from sklearn.pipeline import Pipeline


def predict(clf2, test_set,param,sel):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    test_set = sel.transform(test_set.values)
    print("begin to make predictions")
    # res = clf2.predict_proba(test_set.values)
    res = clf2.predict_proba(test_set)
    uid["proba1"] = pd.Series(res[:, 1])
    uid["score"] = uid.groupby(by=["user_id"])["proba1"].transform(lambda x: sum(x) / float(len(x)))
    uid.drop_duplicates(subset=["user_id"],inplace=True)
    uid.sort_values(by=["score"],axis=0,ascending=False,inplace=True)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    uid_file = "../result/uid/B/uid_lr_" +param+"_"+ str_time + ".csv"
    uid.to_csv(uid_file,header=True,index=False)
    # active_users = uid.loc[uid["score"]>0.5]["user_id"].unique().tolist()
    active_users = uid["user_id"][:24500].unique().tolist()
    # print(len(active_users))
    print(uid["score"].tolist()[24500])
    # print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission_file = "../result/628/pm/submission_lr_" +param+"_"+ str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])
# using this module ,one needs to deconstruct some of the features in data_process
def run(scheme_num=3,file_name="../data/data_v8/training_r"):
    train_set_ls = []
    if scheme_num ==1:
        for i in [16,17,22,23]:
            print("begin to load the dataset")
            file_name1 = file_name+"ld1-"+str(i)+".csv"
            train_set_temp = pd.read_csv(file_name1, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    elif scheme_num ==2:
        for i in [16,23]:
            print("begin to load the dataset")
            file_name2 = file_name+"ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name2, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    elif scheme_num ==3:
        for i in [17,18,19,20,21,22,23]:
            print("begin to load the dataset"+str(i))
            file_name3 = file_name+ "ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name3, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    val_file_name = file_name+ "ld1-23.csv"
    val_set = pd.read_csv(val_file_name, header=0, index_col=None)
    val_set2 =  pd.read_csv("../data/data_v5/training_eld1-23.csv", header=0, index_col=None)
    print(val_set.describe())
    print(val_set2.describe())
    train_set = pd.concat(train_set_ls, axis=0)
    ds = train_set.describe()
    print(ds)

    keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id", "label"]))

    print("begin to drop the duplicates")
    train_set.drop_duplicates(subset=keep_feature, inplace=True)
    val_set.drop_duplicates(subset=keep_feature,inplace=True)
    val_set2.drop_duplicates(subset=keep_feature,inplace=True)
    print(train_set.describe())
    print(val_set.describe())
    print(val_set2.describe())
    train_label = train_set["label"]
    val_label = val_set["label"]
    val_label2 = val_set2["label"]
    train_set = train_set.drop(labels=["label", "user_id"], axis=1)
    val_set = val_set.drop(labels=["label","user_id"], axis=1)
    val_set2 = val_set2.drop(labels=["label","user_id"], axis=1)

    drop_features = [""]


    print("begin to standardization the data")
    for fea in keep_feature:
        train_set[fea] = (train_set[fea]-train_set[fea].min())/(train_set[fea].max()-train_set[fea].min())
        # train_set[fea] = (train_set[fea]-train_set[fea].mean())/(train_set[fea].std())
        val_set[fea] = (val_set[fea]-val_set[fea].min())/(val_set[fea].max()-val_set[fea].min())
        val_set2[fea] = (val_set2[fea]-val_set2[fea].min())/(val_set2[fea].max()-val_set2[fea].min())
        # val_set[fea] = (val_set[fea]-val_set[fea].mean())/(val_set[fea].std())
    # print(train_set.describe())
    # keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id", "label"]))
    # sel = SelectKBest(mutual_info_classif, k=300).fit(train_set.values, train_label.values)
    # train_set = sel.transform(train_set.values)
    # val_set = sel.transform(val_set.values)
    # val_set2 = sel.transform(val_set2.values)
    # feature_importances = sel.scores_
    # print(feature_importances)
    # print(keep_feature)
    # feature_score_name = sorted(zip(feature_importances, keep_feature), reverse=True)
    # for score, name in feature_score_name:
    #     print('{}: {}'.format(name, score))

    # kpca = PCA(n_components=0.98)
    # # kpca = FactorAnalysis(n_components=100)
    # # kpca = KernelPCA(n_components=None,kernel="linear",copy_X=False,n_jobs=-1)
    # kpca.fit(train_set.values)
    # train_set = kpca.transform(train_set.values)
    # val_set = kpca.transform(val_set.values)
    # print(kpca.components_)
    # # # print("eigenvalues of the centered kernel matrix {}".format(kpca.lambdas_))
    # print("number of components {}".format(kpca.n_components_))
    # print("noise variance {}".format(kpca.noise_variance_))
    # print("the explained variance {}".format(kpca.explained_variance_))
    # print("the explained variance ratio {}".format(kpca.explained_variance_ratio_))

    print("begin to make prediction with plain features and without tuning parameters")

    # train_data = lightgbm.Dataset(train_set.values, label=train_label.values, feature_name=list(train_set.columns))

    # best_f1 =0.0
    # best_params = {"n_estimators":800,"num_leaves":6}
    # for n_estimator in [400,600,800]:
    #     for num_leave in [4,6,8]:
    #         print({"n_estimators":n_estimator,"num_leaves":num_leave,"boosting_type":"dart"})
    #         clf1 = LGBMClassifier(n_estimators=n_estimator, num_leaves=num_leave, boosting_type="dart")
    #         clf1.fit(train_set.values, train_label.values)
    #         print("load the test dataset")
    #         yhat = clf1.predict(val_set.values)
    #         print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))
    #         f1 = f1_score(y_pred=yhat, y_true=val_label.values)
    #         if best_f1<f1:
    #             best_f1 = f1
    #             best_params = {"n_estimators":n_estimator,"num_leaves":num_leave,"boosting_type":"dart"}
    # scoring = {'f1': "f1"}
    # clf1 = GridSearchCV(LGBMClassifier(),
    #                   param_grid={"n_estimators":[200,400,600],"num_leaves": [4,5,6,8],"boosting_type":["dart"]},
    #                   scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=1)

    # print({"n_estimators":n_estimator,"num_leaves":num_leave,"boosting_type":"dart"})
    # clf1 = LGBMClassifier(n_estimators=n_estimator, num_leaves=num_leave, boosting_type="dart")
    # clf1 = LogisticRegressionCV(random_state=627,n_jobs=-1,scoring="f1",cv=3,verbose=2,max_iter=1200)
    clf0 = LogisticRegression(random_state=627,verbose=2,max_iter=200,solver="liblinear",penalty="l1")
    clf2 = LogisticRegression(random_state=627,n_jobs=-1,verbose=2,max_iter=200)
    clf1 = Pipeline([
        ('feature_selection', SelectFromModel(clf0)),
        ('classification', clf2)])
    clf1.fit(train_set.values, train_label.values)
    # sel = SelectFromModel(clf1)
    # sel.fit(train_set.values, train_label.values)
    # print("begin to get important features")
    # feature_names = train_set.columns
    # feature_importances = np.reshape(clf1.best_estimator_.coef_, -1)
    # feature_importances = np.reshape(clf1.coef_, -1)
    # # print(feature_importances)
    # # print(feature_names)
    # feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
    # for score, name in feature_score_name:
    #     print('{}: {}'.format(name, score))
    # sorted_feature_name = [name for score, name in feature_score_name]
    # print(sorted_feature_name)


    # clf1.fit(train_set, train_label.values)
    # clf1.fit(train_set.values, train_label.values,eval_set=(val_set.values,val_label.values),early_stopping_rounds=30)
    # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
    # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
    # bs = clf1.best_score_
    # print(bs)
    # bp = clf1.best_params_
    # print(bp)

    print("begin to make classification report for the validation dataset")
    yhat = clf1.predict(val_set.values)
    # yhat = clf1.predict(val_set.values)
    # yhat = clf1.predict(val_set)
    # yhat = clf1.predict(val_set)
    print(classification_report(y_pred=yhat, y_true=val_label.values,digits=4))

    print("begin to make classification report for the original validation dataset")
    # yhat = clf1.predict(val_set.values)
    yhat = clf1.predict(val_set2.values)
    # yhat = clf1.predict(val_set2)
    # val_set2 = sel.transform(val_set2.values)
    # yhat = sel.estimator_.predict(val_set2)
    print(classification_report(y_pred=yhat, y_true=val_label2.values,digits=4))

    print("begin to make classification report for the training dataset")
    yhat = clf1.predict(train_set.values)
    # yhat = clf1.predict(train_set)
    # train_set = sel.transform(train_set.values)
    # yhat = sel.estimator_.predict(train_set)
    print(classification_report(y_pred=yhat, y_true=train_label.values,digits=4))

    print("load the test dataset")
    test_file_name = file_name.replace("training","testing")+ "ld1-30.csv"
    test_set = pd.read_csv(test_file_name,header=0,index_col=None,usecols=keep_feature+["user_id"])
    # # test_set = pd.read_csv("data/testing_rld1-30.csv",header=0,index_col=None)

    for fea in keep_feature:
        test_set[fea] = (test_set[fea]-test_set[fea].min())/(test_set[fea].max()-test_set[fea].min())
        # test_set[fea] = (test_set[fea]-test_set[fea].mean())/(test_set[fea].std())

    print("begin to make prediction")
    param = list(file_name)[-1]+str(scheme_num)
    print(param)
    # predict(clf1,test_set,param)
    # print(clf1.coef_)
    # print(clf1.scores_)
    # predict(clf1,test_set,param,sel)

            # # f1 = f1_score(y_pred=yhat, y_true=val_label.values)
            # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
            # print("begin to get important features")
            # feature_names = train_set.columns
            # # feature_importances = clf1.best_estimator_.feature_importances_
            # # feature_importances = clf1.feature_importances_
            # feature_importances = clf1.feature_importances_
            # print(feature_importances)
            # print(feature_names)
            # feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
            # for score, name in feature_score_name:
            #     print('{}: {}'.format(name, score))
            # sorted_feature_name = [name for score, name in feature_score_name]
            # print(sorted_feature_name)
            # with open("kuaishou_stats.csv", 'a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(["feature importance of lgb for kuaishou ", str_time])
            #     # writer.writerow(["best score", bs, "best params"])
            #     # for key, value in bp.items():
            #     #     writer.writerow([key, value])
            #     # writer.writerow(eval_metrics)
            #     feature_score_name = sorted(zip(feature_importances, feature_names), reverse=True)
            #     for score, name in feature_score_name:
            #         # print('{}: {}'.format(name, score))
            #         writer.writerow([name, score])

if __name__ == "__main__":
    file_name1 = "../data/data_v9/training_e"
    file_name2 = "../data/data_v8/training_r"
    for scheme in [3]:
        for file in [file_name1]:
            run(scheme_num=scheme,file_name=file)