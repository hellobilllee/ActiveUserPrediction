import csv
import datetime
import numpy as np
import hyperopt
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.decomposition import  PCA
from sklearn.metrics import classification_report

def predict(clf2, test_set, param, kpca):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    test_set = kpca.transform(test_set.values)
    print("begin to make predictions")
    # res = clf2.predict_proba(test_set.values)
    res = clf2.predict_proba(test_set)
    uid["proba1"] = pd.Series(res[:, 1])
    uid["score"] = uid.groupby(by=["user_id"])["proba1"].transform(lambda x: sum(x) / float(len(x)))
    uid.drop_duplicates(subset=["user_id"], inplace=True)
    uid.sort_values(by=["score"], axis=0, ascending=False, inplace=True)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    uid_file = "../result/uid/uid_cb_" + param + "_" + str_time + ".csv"
    uid.to_csv(uid_file, header=True, index=False)
    # active_users = uid.loc[uid["score"]>0.5]["user_id"].unique().tolist()
    active_users = uid["user_id"][:24500].unique().tolist()
    # print(len(active_users))
    print(uid["score"].tolist()[24500])
    # print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission_file = "../result/622/submission_cb_" + param + "_" + str_time + ".csv"
    with open(submission_file, "a", newline="") as f:
        writer = csv.writer(f)
        for i in active_users:
            writer.writerow([i])


# using this module ,one needs to deconstruct some of the features in data_process
def run(scheme_num=1, file_name="../data/data_v3/training_e"):
    train_set_ls = []
    if scheme_num == 1:
        for i in [16, 17, 22, 23]:
            print("begin to load the dataset")
            file_name1 = file_name + "ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name1, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    elif scheme_num == 2:
        for i in [16, 23]:
            print("begin to load the dataset")
            file_name2 = file_name + "ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name2, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    elif scheme_num == 3:
        for i in [18, 19, 20, 21, 22, 23]:
            print("begin to load the dataset")
            file_name3 = file_name + "ld1-" + str(i) + ".csv"
            train_set_temp = pd.read_csv(file_name3, header=0, index_col=None)
            print(train_set_temp.describe())
            train_set_ls.append(train_set_temp)
    val_file_name = file_name + "ld1-23.csv"
    val_set = pd.read_csv(val_file_name, header=0, index_col=None)
    print(val_set.describe())
    train_set = pd.concat(train_set_ls, axis=0)
    ds = train_set.describe()
    print(ds)
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

    print("begin to standardization the data")
    for fea in keep_feature:
        if train_set[fea].var() < 0.000001 or val_set[fea].var() < 0.000001:
            train_set.drop(labels=[fea], axis=1, inplace=True)
            val_set.drop(labels=[fea], axis=1, inplace=True)
        else:
            train_set[fea] = (train_set[fea] - train_set[fea].min()) / (train_set[fea].max() - train_set[fea].min())
            # train_set[fea] = (train_set[fea]-train_set[fea].mean())/(train_set[fea].std())
            val_set[fea] = (val_set[fea] - val_set[fea].min()) / (val_set[fea].max() - val_set[fea].min())
            # val_set[fea] = (val_set[fea]-val_set[fea].mean())/(val_set[fea].std())
    keep_feature = list(set(train_set.columns.values.tolist()) - set(["user_id", "label"]))
    kpca = PCA(n_components=0.99, whiten=True)
    # # kpca = KernelPCA(n_components=None,kernel="linear",copy_X=False,n_jobs=-1)
    kpca.fit(train_set.values)
    train_set = kpca.transform(train_set.values)
    val_set = kpca.transform(val_set.values)
    # # print("eigenvalues of the centered kernel matrix {}".format(kpca.lambdas_))
    print("number of components {}".format(kpca.n_components_))
    print("noise variance {}".format(kpca.noise_variance_))
    print("the explained variance {}".format(kpca.explained_variance_))
    print("the explained variance ratio {}".format(kpca.explained_variance_ratio_))

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
    scoring = {'f1': "f1"}
    # clf1 = GridSearchCV(LGBMClassifier(),
    #                   param_grid={"n_estimators":[200,400,600],"num_leaves": [4,5,6,8],"boosting_type":["dart"]},
    #                   scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=1)
    # for n_estimator in [200,500,800]:
    #     for depth in [6]:
    #         print({"n_estimators": n_estimator, "depth": depth})
    #         clf1 = CatBoostClassifier(iterations=n_estimator, depth=depth,verbose=2)
    #         # clf1.fit(train_set.values, train_label.values)
    #         clf1.fit(train_set, train_label.values)
    #         # clf1.fit(train_set.values, train_label.values,eval_set=(val_set.values,val_label.values),early_stopping_rounds=30)
    #         # cv_results = cv(initial_params,train_data,num_boost_round=800,nfold=4,early_stopping_rounds=30,verbose_eval=True)
    #         # bst = lgb.cv(initial_params, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)
    #         # bs = clf1.best_score_
    #         # print(bs)
    #         # bp = clf1.best_params_
    #         # print(bp)
    #
    #         print("begin to make classification report for the validation dataset")
    #         # yhat = clf1.predict(val_set.values)
    #         # yhat = clf1.predict(val_set.values)
    #         yhat = clf1.predict(val_set)
    #         print(classification_report(y_pred=yhat, y_true=val_label.values, digits=4))
    #
    #         print("begin to make classification report for the training dataset")
    #         # yhat = clf1.predict(train_set.values)
    #         yhat = clf1.predict(train_set)
    #         print(classification_report(y_pred=yhat, y_true=train_label.values, digits=4))
    #
    #         print("load the test dataset")
    #         test_file_name = file_name.replace("training", "testing") + "ld1-30.csv"
    #         test_set = pd.read_csv(test_file_name, header=0, index_col=None, usecols=keep_feature + ["user_id"])
    #         # test_set = pd.read_csv("data/testing_rld1-30.csv",header=0,index_col=None)
    #         for fea in keep_feature:
    #             test_set[fea] = (test_set[fea] - test_set[fea].min()) / (test_set[fea].max() - test_set[fea].min())
    #             # test_set[fea] = (test_set[fea]-test_set[fea].mean())/(test_set[fea].std())

    print("begin to tune the parameters ")
    paramsSpace = {
        "n_estimators":hyperopt.hp.quniform("n_estimators", 500, 1200, 100),
        'depth': hyperopt.hp.quniform("depth", 6, 8, 1),
        "border_count": hyperopt.hp.quniform("border_count", 128, 148, 4),
        'learning_rate': hyperopt.hp.loguniform('learning_rate', 1e-4, 1e-1),
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 1, 48,2),
        'bagging_temperature': hyperopt.hp.uniform('bagging_temperature', 0.9, 1.0),
        'rsm': hyperopt.hp.uniform('rsm', 0.8, 1.0),
        # "leaf_estimation_method": hyperopt.hp.choice("leaf_estimation_method",['Newton', 'Gradient']),
    }
    #
    # # train_x, val_x, train_y, val_y = train_test_split(train_set.values, train_label.values, test_size=0.33,
    # #                                                   random_state=42)
    def hyperopt_objective(params):
        model = CatBoostClassifier(
            n_estimators=params["n_estimators"],
            # use_best_model=True,od_type='Iter',od_wait=20,
            verbose=2,
            eval_metric='Logloss',
            od_pval=0.000001,
            # leaf_estimation_method=params['leaf_estimation_method'],
            depth=params['depth'],
            border_count=params['border_count'],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params['l2_leaf_reg'],bagging_temperature=params['bagging_temperature'],
            rsm=params['rsm'])
        cv_data = cv(Pool(train_set,train_label),model.get_params(),nfold=3,verbose_eval=True)
        # model.fit(train_pool_tp, eval_set=validate_pool_tp)
        # model.fit(X=train_x, y=train_y,
        #         eval_set=(val_x, val_y))
        # y_val_hat = model.predict(train_set.values)
        # mean_auc = roc_auc_score(train_label.values, y_val_hat)
        # metrics = model.eval_metrics(validate_pool_tf, ['AUC'])
        # mean_auc = sum(metrics['AUC'])/float(len(metrics['AUC']))
        # cv_data = cv(
        #     Pool(train_set_tf, train_label, cat_features=categorical_features_indices_tf),
        #     model.get_params()
        # )
        logloss = np.max(cv_data['test-Logloss-mean'])
        print(logloss)
        return logloss  # as hyperopt minimises
    best_params = hyperopt.fmin(
        hyperopt_objective,
        space=paramsSpace,
        algo=hyperopt.tpe.suggest,
        max_evals=100,
    )
    print(best_params)
    clf1 = CatBoostClassifier(
        verbose=2,loss_function="Logloss",
        iterations=best_params["n_estimators"],
        eval_metric="Logloss",
        custom_metric="Logloss",
        random_seed=42,
        # use_best_model=True,
        # od_type='Iter',od_wait=20,
        # leaf_estimation_method=best_params['leaf_estimation_method'],
        depth=best_params['depth'],
        border_count=best_params['border_count'],
        learning_rate=best_params["learning_rate"],l2_leaf_reg=best_params['l2_leaf_reg'],
        bagging_temperature=best_params['bagging_temperature'],rsm=best_params['rsm'])
    # cv_data = cv(Pool(train_set.values, train_label.values), clf2.get_params(),nfold=5)
    # clf2.fit(X=train_x, y=train_y,
    #           eval_set=(val_x, val_y))
    # print(cv_data)
    clf1.fit(train_set,train_label.values)
    print("parameter tuning over, begin to save the model!")
    # str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # model_name = "catboost_" + str_time + ".pkl"
    # clf2.save_model(model_name)
    # # joblib.dump(clf2, model_name)
    print("begin to make classification report for the training dataset")
    # yhat = clf1.predict(train_set.values)
    yhat = clf1.predict(train_set)
    print(classification_report(y_pred=yhat, y_true=train_label.values, digits=4))

    print("load the test dataset")
    test_file_name = file_name.replace("training", "testing") + "ld1-30.csv"
    test_set = pd.read_csv(test_file_name, header=0, index_col=None, usecols=keep_feature + ["user_id"])
    # test_set = pd.read_csv("data/testing_rld1-30.csv",header=0,index_col=None)
    for fea in keep_feature:
        test_set[fea] = (test_set[fea] - test_set[fea].min()) / (test_set[fea].max() - test_set[fea].min())
        # test_set[fea] = (test_set[fea]-test_set[fea].mean())/(test_set[fea].std())

    print("begin to make prediction")
    param = list(file_name)[-1] + str(scheme_num) + "_tuned"
    print(param)
    # predict(clf1,test_set,param)
    predict(clf1, test_set, param, kpca)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    print(str_time)


if __name__ == "__main__":
    file_name1 = "../data/data_v3/training_e"
    file_name2 = "../data/data_v4/training_r"
    for scheme in [3]:
        for file in ["../data/data_v4/training_r"]:
            run(scheme_num=scheme,file_name=file)