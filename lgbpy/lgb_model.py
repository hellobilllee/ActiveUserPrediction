
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

def LR_test(train_x,train_y,test_x,test_y):
    from sklearn.metrics import roc_auc_score

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.06, random_state=None,

                            solver='liblinear', max_iter=100,

                            verbose=1)
    lr.fit(train_x,train_y)

    predict = lr.predict_proba(test_x)[:, 1]

    print(roc_auc_score(test_y,predict))
def CAT_test(train_x,train_y, val_x,val_y):
    import pandas as pd
    initial_params = {
        "verbose": 100,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "random_seed": 42,
        "learning_rate": 0.02,
        # "one_hot_max_size": 2,
        "depth": 6,
        # "border_count": 128,
        "thread_count": 16,
        # "class_weights":[0.1,1.8],
        # "l2_leaf_reg": 6,
        "use_best_model": True,
        # "save_snapshot":True,
        # "leaf_estimation_method": 'Newton',
        "od_type": 'Iter',
        "od_wait": 30,
        # "od_pval":0.0000001,
        # "used_ram_limit":1024*1024*1024*12,
        # "max_ctr_complexity":3,
        # "model_size_reg":10,
    }
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(**initial_params)
    clf.fit(X=train_x, y=train_y, eval_set=(val_x, val_y),verbose_eval=100)
    feature_importances = sorted(zip(train_x.columns, clf.feature_importances_), key=lambda x: x[1],reverse=True)
    feature_importances = pd.DataFrame([list(f) for f in feature_importances], columns=["features", "importance"])
    return clf.score(val_x,val_y),feature_importances
def xgb_test(train_x,train_y, val_x,val_y):
    import pandas as pd
    import xgboost as xgb
    initial_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 3000,
        "booster":"gbtree",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_jobs": -1,
        "colsample_bytree": 0.8,
        "subsample": 0.7,
        "min_child_weight": 1,
        "reg_lambda": 1,
    }
    clf = xgb.XGBClassifier(**initial_params)
    clf.fit(X=train_x, y=train_y, eval_set=[(train_x,train_y),(val_x,val_y)],eval_metric="auc",verbose=True,early_stopping_rounds=100)
    feature_importances_gain = sorted(zip(train_x.columns, clf.feature_importances_),
                                      key=lambda x: x[1], reverse=True)
    feature_importances_gain = pd.DataFrame([list(f) for f in feature_importances_gain], columns=["features", "importance"])

    return feature_importances_gain
def rf_test(train_x,train_y, val_x,val_y):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=50,
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
    )
    clf.fit(X=train_x, y=train_y)
    predict = clf.predict_proba(val_x)[:, 1]

    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(val_y,predict))
    feature_importances = sorted(zip(train_x.columns, clf.feature_importances_),
                                      key=lambda x: x[1], reverse=True)
    feature_importances = pd.DataFrame([list(f) for f in feature_importances], columns=["features", "importance"])

    return feature_importances
def lgb_test_gain(train_x,train_y, val_x,val_y):
    print("LGB test")
    d_train = lgb.Dataset(train_x, label=train_y)
    d_test = lgb.Dataset(val_x, label=val_y)
    params = {
        # "max_bin": 512,
        "learning_rate": 0.01,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "max_depth": -1,
        "verbose": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "subsample_freq": 1,
        "reg_alpha": 0,
        "min_child_weight": 25,
        "random_state": 2018,
        "reg_lambda": 1,
        "n_jobs": -1,
    }
    print("begin to train ")
    clf = lgb.train(params, d_train, 3000, valid_sets=[d_test], early_stopping_rounds=100, verbose_eval=100)
    feature_importances_gain = sorted(zip(train_x.columns, clf.feature_importance(importance_type="gain")),
                                      key=lambda x: x[1], reverse=True)
    feature_importances_gain = pd.DataFrame([list(f) for f in feature_importances_gain], columns=["features", "importance"])
    return feature_importances_gain
def lgb_test_sk(train_x,train_y,test_x,test_y,ftw="gain"):
    print("LGB test")

    clf = lgb.LGBMClassifier(

        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,

        max_depth=-1, n_estimators=3000, objective='binary',

        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,

        learning_rate=0.01, min_child_weight=25,random_state=2018,n_jobs=50

    )
    clf.fit(train_x, train_y,eval_set=[(train_x,train_y),(test_x,test_y)],eval_metric="auc",early_stopping_rounds=100,verbose=0)
    if ftw=="split":
        feature_importances=sorted(zip(train_x.columns,clf.feature_importances_),key=lambda x:x[1],reverse=True)
        feature_importances = pd.DataFrame([list(f) for f in feature_importances], columns=["features", "importance"])
    elif ftw=="shap":
        explainer = shap.TreeExplainer(clf)
        shap_sample = test_x.sample(frac=1.0)
        shap_values = explainer.shap_values(shap_sample)
        shap.summary_plot(shap_values, shap_sample)
        shap.summary_plot(shap_values, shap_sample, plot_type="bar")
        feature_importances_shap = sorted(zip(train_x.columns, np.mean(np.abs(shap_values), axis=0)),
                                          key=lambda x: x[1], reverse=True)
        feature_importances = pd.DataFrame([list(f) for f in feature_importances_shap],
                                                columns=["features", "importance"])
    return clf.best_score_[ 'valid_1']['auc'],feature_importances

def lgb_tune_btb(train_x, train_y, val_x, val_y,n_turn=30,verbose=True):
    from btb.tuning import GP
    from btb import HyperParameter, ParamTypes
    tunables = [
        # ('n_estimators', HyperParameter(ParamTypes.INT, [10, 500])),
                ('num_leaves', HyperParameter(ParamTypes.INT, [28, 64])),
                ("learning_rate", HyperParameter(ParamTypes.FLOAT, [0.01, 0.05])),
                ("colsample_bytree", HyperParameter(ParamTypes.FLOAT, [0.6, 1.0])),
                ("subsample", HyperParameter(ParamTypes.FLOAT, [0.6, 1.0])),
                ("reg_alpha", HyperParameter(ParamTypes.INT, [0, 32])),
                ("reg_lambda", HyperParameter(ParamTypes.INT, [0, 64])),
                ("min_child_weight", HyperParameter(ParamTypes.INT, [1, 32])),
                # ("max_bin", HyperParameter(ParamTypes.INT, [256, 512])),
                ]
    tuner = GP(tunables)
    def tune_lgb(tuner,train_x, train_y, val_x, val_y,n_turn):
        param_ls = []
        score_ls = []
        for i in range(n_turn):
            print("the {}th round ".format(i))
            params = tuner.propose()
            params.update({"boosting_type":'gbdt',"n_estimators":4000,"n_jobs":-1,"objective":'binary',"metric": "auc","max_depth": -1})
            d_train = lgb.Dataset(train_x, label=train_y)
            d_test = lgb.Dataset(val_x, label=val_y)
            model = lgb.train(params, d_train, 3000, valid_sets=[d_train, d_test], early_stopping_rounds=100,
                            verbose_eval=200)
            # model = lgb.LGBMClassifier(
            #     boosting_type='gbdt',
            #     n_estimators=4000,
            #     n_jobs=-1,
            #     objective='binary',
            #     min_child_weight=params['min_child_weight'],
            #     verbose=200, eval_metric='auc',
            #     num_leaves=params['num_leaves'],
            #     learning_rate=params["learning_rate"],
            #     reg_alpha=params["reg_alpha"],
            #     reg_lambda=params["reg_lambda"],
            #     subsample=params["subsample"],
            #     colsample_bytree=params["colsample_bytree"])
            # model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], eval_metric="auc",
            #           early_stopping_rounds=100,verbose=200)
            auc = model.best_score["valid_1"]["auc"]
            best_n_estimator = model.best_iteration
            params.update({"n_estimators":best_n_estimator})
            if verbose:
                print("params:", params)
                print("validation auc:", auc)
            param_ls.append(params)
            score_ls.append(auc)
            tuner.add(params, auc)
            del d_train,d_test,model
            import gc
            gc.collect
        best_params = param_ls[score_ls.index(max(score_ls))]
        if verbose:
            print("best params:", best_params)
            print("best score:", tuner._best_score)
        return best_params
    return tune_lgb(tuner,train_x, train_y, val_x, val_y,n_turn)
def lgb_tune_opt(train_x, train_y, val_x, val_y):
    import hyperopt
    print("begin to tune the parameters ")
    paramsSpace = {
        # "n_estimators":hyperopt.hp.quniform("n_estimators", 800, 3000, 100),
        "num_leaves": hyperopt.hp.quniform("num_leaves", 4, 64, 3),
        "min_child_weight": hyperopt.hp.quniform("min_child_weight", 0,32, 2),
        'learning_rate': hyperopt.hp.loguniform('learning_rate', 1e-3, 1e-1),
        'reg_alpha': hyperopt.hp.quniform('reg_alpha', 0, 32,1),
        'reg_lambda': hyperopt.hp.quniform('reg_lambda', 0, 64,2),
        # 'subsample_freq': hyperopt.hp.uniform('subsample_freq', 0.9, 1.0),
        'subsample': hyperopt.hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.7, 1.0),
    }
    def hyperopt_objective(params):
        # clf = lgb.train(params, d_train, 3000, valid_sets=[d_test], early_stopping_rounds=100, verbose_eval=100)
        model = lgb.LGBMClassifier(
            verbose=100,eval_metric='auc',
            num_leaves=params['num_leaves'],
            min_child_weight=params['min_child_weight'],
            learning_rate=params["learning_rate"],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'])
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)], eval_metric="auc",early_stopping_rounds = 100)
        auc = model.best_score_['valid_1']['auc']
        print("validation auc {}".format(str(auc)))
        return 1-auc # as hyperopt minimises
    best_params = hyperopt.fmin(
        fn=hyperopt_objective,
        space=paramsSpace,
        algo=hyperopt.tpe.suggest,
        max_evals=60,
    )
    print(best_params)
def stacking_test():
    import lightgbm as lgb
    import xgboost as xgb
    clf_lgb = lgb.LGBMClassifier(

        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,

        max_depth=-1, n_estimators=3000, objective='binary',

        subsample=0.7, colsample_bytree=0.8, subsample_freq=1,  # colsample_bylevel=0.7,

        learning_rate=0.01, min_child_weight=25,random_state=2018,n_jobs=50

    )

    clf_xgb = xgb.XGBClassifier(objective="binary:logistic",n_estimators=3000,booster="gbtree",learning_rate=0.05
                                ,max_depth=7,n_jobs=-1,colsample_bytree=0.7,subsample=0.7,min_child_weight=1,reg_lambda=1)
    from sklearn.ensemble import RandomForestClassifier
    clf_rf = RandomForestClassifier(
        n_estimators=1200,
        criterion='gini',
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        min_impurity_decrease=0.0,
        n_jobs=-1,
        random_state=0,
        warm_start=False
    )
    cat_params = {
        "verbose": 100,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "random_seed": 42,
        "learning_rate": 0.02,
        "depth": 6,
        "thread_count": 50,
        "use_best_model": True,
        "od_type": 'Iter',
        "od_wait": 30,
    }
    from catboost import CatBoostClassifier
    clf_cat = CatBoostClassifier(**cat_params)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    from mlxtend.classifier import StackingClassifier
    sclf = StackingClassifier(classifiers=[clf_lgb, clf_xgb,clf_cat,clf_rf],
                              meta_classifier=lr)


def get_stratified_sample(df, sample_target="user_id", reference_target="app_launch_day",
                          sample_ratio=0.2):
    df = df.astype(np.uint32)
    reference_target_ls = df[reference_target].unique().tolist()
    target_sample = []
    for i in reference_target_ls:
        # print("get users in day {}".format(i))
        target_sample.extend(df.loc[df[reference_target] == int(i)][sample_target].drop_duplicates().sample(frac=sample_ratio).tolist())
    del df
    return list(set(target_sample))
def get_normalized_rank(df):
    df.reset_index(drop=True,inplace=True)
    df["order"] = df.index
    df.sort_values(by="prob",axis=0,ascending=True,inplace=True)
    df.reset_index(drop=True,inplace=True)
    df["rank"] =df.index
    df.sort_values(by="order", axis=0, ascending=True, inplace=True)
    df.reset_index(drop=True,inplace=True)
    df["score"] = (df["rank"]-df["rank"].min())/(df["rank"].max()-df["rank"].min())
    return df[["user_id","score"]]
def lgb_predict(train_set,val_set,test_set,file,best_params=None,val_ratio=0.4, n_round = 3):
    import lightgbm as lgb
    import numpy as np
    from scipy.stats import rankdata
    import random
    import gc
    res=test_set[['user_id']]
    test_x = test_set.drop(['user_id'], axis=1)
    res['prob'] = 0
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": np.uint8}
    testb = \
    pd.read_table('/mnt/datasets/fusai/user_register_log.txt', header=None, names=user_register_log, index_col=None,
                  dtype=dtype_user_register)[['user_id']]
    if not best_params:
        best_params = {
            # "max_bin": 512,
            "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "max_depth": -1,
            # "verbose": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "subsample_freq": 1,
            "reg_alpha": 0,
            "min_child_weight": 25,
            "random_state": 2018,
            "reg_lambda": 1,
            "n_jobs": -1,
        }
    print("begin to train ")

    # val_len = int(len(val_set)*0.4)
    use_cols = list(set(train_set.columns)-set(["user_id","label"]))
    df_val_user = pd.read_pickle("../work/val_user_17_23.pkl")
    val_user_all = df_val_user["user_id"].unique().tolist()
    final_rank = 0
    train_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    val_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")

    for i in range(n_round):
        random.seed(np.random.randint(1, 1000))
        # val = val_set.iloc[-val_len:, :].sample(frac=val_ratio)
        # val = val_set.sample(frac=val_ratio)
        val_user = get_stratified_sample(df_val_user,sample_ratio=val_ratio)
        val_user_add = list(set(val_user_all)-set(val_user))
        val = val_set[val_set["user_id"].isin(val_user)]
        # val_add = val_set[val_set["user_id"].isin(val_user_add)]
        val_train = val_set[~val_set["user_id"].isin(val["user_id"])]
        train = pd.concat([train_set, val_train], axis=0)
        # train = pd.concat([train_set, val_train,val_add], axis=0)
        print("the {}th round".format(i))
        print("shape of val:", val.shape)
        print("shape of train:", train.shape)
        train_y = train['label']
        train_x = train.drop(['user_id', "label"], axis=1)
        val_y = val['label']
        val_x = val.drop(['user_id', "label"], axis=1)
        d_train = lgb.Dataset(train_x, label=train_y)
        d_test = lgb.Dataset(val_x, label=val_y)
        best_params.update({"random_state": 2018+i})
        clf_lgb = lgb.train(best_params, d_train, 5000, valid_sets=[d_train, d_test], early_stopping_rounds=100,
                        verbose_eval=200)
        temp_score_val = clf_lgb.best_score["valid_1"]["auc"]
        temp_score_train = clf_lgb.best_score["training"]["auc"]
        # weight = 2*(temp_score_train*temp_score_val)/(temp_score_train+temp_score_val)
        weight = (temp_score_train+2*temp_score_val)
        res_temp = test_set[['user_id']]
        res_temp['prob'] = 0
        temp_predict = clf_lgb.predict(test_x,num_iteration=clf_lgb.best_iteration)
        res_temp['prob'] = temp_predict
        res_temp = pd.merge(testb, res_temp, on='user_id', how='left').fillna(0)
        res_temp.to_csv('../input/' + file +str(i)+ '.txt', sep=',', index=False, header=False)
        # res_temp = get_normalized_rank(res_temp)
        # res['prob']+= res_temp['rank']
        # res['prob']+= res_temp['rank']/n_round
        # res['prob']+= res_temp['prob']/n_round
        res['prob']+= temp_predict*temp_predict* weight/n_round
        # res_temp = res_temp[["user_id","rank"]]
        # final_rank = final_rank+rankdata(temp_predict, method='ordinal') * weight
        del val, val_train,train,train_y,train_x,val_y,val_x,d_train,d_test,clf_lgb,res_temp,temp_predict
        gc.collect()
    # res["prob"] = (final_rank -min(final_rank))/(max(final_rank)-min(final_rank))
    # res["prob"] = (res["prob"] -min(res["prob"]))/(max(res["prob"])-min(res["prob"]))
    res=pd.merge(testb,res,on='user_id',how='left').fillna(0)
    res.to_csv('../work/' + file + '.txt', sep=',', index=False,header=False)
    del testb,train_set, val_set,test_set
    gc.collect()
    return res

