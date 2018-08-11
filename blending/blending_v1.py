import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from functools import partial

def one_minus_roc(X, y, est):
    pred = est.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score
    return 1 - roc_auc_score(y, pred)

class EarlyStoppingSK(ClassifierMixin):
    def __init__(self, estimator, max_n_estimators, scorer,
                 n_min_iterations=100, scale=1.0001,early_stopping_rounds=10):
        self.estimator = estimator
        self.max_n_estimators = max_n_estimators
        self.scorer = scorer
        self.scale = scale
        self.n_min_iterations = n_min_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.best_score = 0.0
        self.best_n_est = 1

    def _make_estimator(self, append=True):
        """Make and configure a copy of the `estimator` attribute.

        Any estimator that has a `warm_start` option will work.
        """
        estimator = clone(self.estimator)
        estimator.n_estimators = 1
        estimator.warm_start = True
        return estimator

    def fit(self, X, y):
        """Fit `estimator` using X and y as training set.

        Fits up to `max_n_estimators` iterations and measures the performance
        on a separate dataset using `scorer`
        """
        est = self._make_estimator()
        self.scores_ = {}
        early_stopping_rounds = 0

        for n_est in range(1, self.max_n_estimators + 1):
            est.n_estimators = n_est
            est.fit(X, y)

            score = self.scorer(est)
            print("{} estimator, validation score {}: ".format(n_est,1-score))
            self.estimator_ = est
            self.scores_.update({n_est:score})

            if (n_est > self.n_min_iterations and
                        score > self.scale * min(self.scores_.values())):
                early_stopping_rounds+=1
                if early_stopping_rounds==self.early_stopping_rounds:
                    self.best_score=1- min(self.scores_.values())
                    self.best_n_est = max(self.scores_.keys())
                    est.n_estimators = self.best_n_est
                    est.fit(X, y)
                    return self

        return self
    def predict(self,X):
        return self.estimator_.predict(X)
    def predict_proba(self,X):
        return self.estimator_.predict_proba(X)
def stop_early_wrapper(classifier, train_x,train_y,val_x,val_y,n_min_iterations=100,scale=1.0001,eval_metric="auc",early_stopping_rounds=10):
    n_iterations = classifier.n_estimators
    if eval_metric=="auc":
        scorer = partial(one_minus_roc, val_x, val_y)
    early = EarlyStoppingSK(classifier,
                          max_n_estimators=n_iterations,
                          # fix the dataset used for testing by currying
                          scorer=scorer,
                          n_min_iterations=n_min_iterations, scale=scale,early_stopping_rounds=early_stopping_rounds)
    early.fit(train_x,train_y)
    return early

def KerasClassifier_wrapper(input_dims):
    import tensorflow as tf
    from keras import backend as K
    # AUC for a binary classifier
    def auc(y_true, y_pred):
        ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
        pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
        pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
        binSizes = -(pfas[1:] - pfas[:-1])
        s = ptas * binSizes
        return K.sum(s, axis=0)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # PFA, prob false alert for binary classifier
    def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(1 - y_true)
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_pred - y_pred * y_true)
        return FP / N

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # P_TA prob true alerts for binary classifier
    def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true)
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)
        return TP / P
        # prepare callbacks
    def model():
        from keras.models import Sequential
        model = Sequential()
        # input layer
        from keras.layers import Dense
        model.add(Dense(input_dims, input_dim=input_dims))
        from keras.layers import BatchNormalization
        model.add(BatchNormalization())
        from keras.layers import Activation
        model.add(Activation('relu'))
        from keras.layers import Dropout
        model.add(Dropout(0.4))
        # hidden layers
        model.add(Dense(input_dims))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(input_dims // 2))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(input_dims // 4, activation='relu'))

        # output layer (y_pred)
        model.add(Dense(1, activation='sigmoid'))

        # compile this model
        model.compile(loss='binary_crossentropy',  # one may use 'mean_absolute_error' as alternative
                      optimizer='adam',
                      metrics=[auc]  # you can add several if needed
                      )
        # Visualize NN architecture
        print(model.summary())
        return model
    from keras.wrappers.scikit_learn import KerasClassifier
    return KerasClassifier(build_fn=model)


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

def blending_test(train_x,train_y,val_x,val_y,test_x,weights=(0.38,0.18,0.14,0.1,0.20)):
    import random
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cat
    from sklearn.ensemble import RandomForestClassifier as rf
    from sklearn.linear_model import LogisticRegression as lr
    import gc
    from scipy.stats import rankdata
    res = 0
    lgb_params = {
            # "max_bin": 512,
            "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "max_depth": -1,
            "verbose": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "subsample_freq": 1,
            "reg_alpha": 0,
            "min_child_weight": 25,
            "random_state": random.randint(1,1000),
            "reg_lambda": 1,
            "n_jobs": -1,
        }
    d_train = lgb.Dataset(train_x, label=train_y)
    d_test = lgb.Dataset(val_x, label=val_y)
    clf_lgb = lgb.train(lgb_params, d_train, 3000, valid_sets=[d_train, d_test], early_stopping_rounds=100,
                        verbose_eval=200)
    # temp_score_val = clf_lgb.best_score["valid_1"]["auc"]
    # temp_score_train = clf_lgb.best_score["training"]["auc"]
    # weight = 2*(temp_score_train*temp_score_val)/(temp_score_train+temp_score_val)
    temp_predict_lgb = clf_lgb.predict(test_x, num_iteration=clf_lgb.best_iteration)
    res = res + rankdata(temp_predict_lgb, method='ordinal')*weights[0]
    # res+=temp_predict_lgb*weights[0]
    # weight = (temp_score_train + 2 * temp_score_val)/3
    # res_temp_lgb = temp_predict_lgb * temp_predict_lgb * weight
    del d_test,d_train,clf_lgb,lgb_params

    cat_params = {
        "verbose": 200,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "random_seed": random.randint(1,1000),
        "learning_rate": 0.03,
        "depth": 6,
        "thread_count": 16,
        "use_best_model": True,
        "od_type": 'Iter',
        "od_wait": 30,
    }
    from catboost import CatBoostClassifier
    clf_cat = CatBoostClassifier(**cat_params)
    clf_cat.fit(X=train_x, y=train_y, eval_set=(val_x, val_y),verbose_eval=200)
    # clf_cat = cat.train(pool=(train_x, train_y),params=cat_params,eval_set=[(train_x, train_y),(val_x, val_y)],verbose_eval=200)
    # temp_score_val = clf_cat.get_test_eval()[1]
    # temp_score_train = clf_cat.get_test_eval()[0]
    temp_predict_cat = clf_cat.predict_proba(test_x)[:,1]
    res = res + rankdata(temp_predict_cat, method='ordinal')*weights[1]
    # res+=temp_predict_cat*weights[1]
    # weight = (temp_score_train + 2 * temp_score_val)/3
    # res_temp_cat = temp_predict_cat * temp_predict_cat * weight
    del clf_cat,cat_params
    gc.collect()
    # xgb_params = {
    #     "objective": "binary:logistic",
    #     "eval_metric": "auc",
    #     "n_estimators": 3000,
    #     "booster":"gbtree",
    #     "learning_rate": 0.05,
    #     "max_depth": 6,
    #     "n_jobs": -1,
    #     "colsample_bytree": 0.9,
    #     "subsample": 0.8,
    #     "min_child_weight": 1,
    #     "reg_lambda": 1,
    # }
    # clf_xgb = xgb.XGBClassifier(**xgb_params)
    # clf_xgb.fit(X=train_x, y=train_y, eval_set=[(train_x,train_y),(val_x,val_y)],eval_metric="auc",verbose=False,early_stopping_rounds=100)
    # temp_score_val = clf_xgb.best_score["valid_1"]["auc"]
    # temp_score_train = clf_xgb.best_score["training"]["auc"]
    # # weight = 2*(temp_score_train*temp_score_val)/(temp_score_train+temp_score_val)
    # temp_predict_xgb = clf_xgb.predict(test_x,ntree_limit=clf_xgb.best_ntree_limit)
    # weight = (temp_score_train + 2 * temp_score_val)/3
    # res_temp_xgb = temp_predict_xgb * temp_predict_xgb * weight
    # del clf_xgb,temp_score_val,temp_score_train,temp_predict_xgb,xgb_params
    rf_ = rf(
        n_estimators=1000,
        criterion='gini',
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=0.6,
        min_impurity_decrease=0.00001,
        n_jobs=-1,
        verbose=200,
        random_state=random.randint(1,1000),
        warm_start=True,
    )
    clf_rf = stop_early_wrapper(rf_,train_x.values,train_y.values,val_x.values, val_y.values,n_min_iterations=10,scale=1.0001,eval_metric="auc",early_stopping_rounds=10)
    temp_predict_rf = clf_rf.predict_proba(test_x.values)[:,1]
    res = res + rankdata(temp_predict_rf, method='ordinal')*weights[2]
    # res+=temp_predict_rf*weights[2]
    # weight = clf_rf.best_score
    # res_temp_rf = temp_predict_rf * temp_predict_rf * weight
    del clf_rf,rf_,
    gc.collect()
    for f in test_x.columns:
        train_x[f] = (train_x[f] - train_x[f].min()) / (train_x[f].max() - train_x[f].min())
        val_x[f] = (val_x[f] - val_x[f].min()) / (val_x[f].max() - val_x[f].min())
        test_x[f] = (test_x[f] - test_x[f].min()) / (test_x[f].max() - test_x[f].min())
    clf_lr = lr(penalty='l2', dual=False, tol=0.00001, C=0.06, random_state=random.randint(1,1000),

                            solver='liblinear', max_iter=2000,

                            verbose=200)
    temp_train = pd.concat([train_x,val_x],axis=0)
    temp_y = pd.concat([train_y,val_y],axis=0)
    clf_lr.fit(temp_train,temp_y)
    print("validation score of lr",clf_lr.score(val_x,val_y))
    # val_y_hat = clf_lr.predict_proba(test_x)[:, 1]
    # from sklearn.metrics import roc_auc_score
    # weight = roc_auc_score(val_y, val_y_hat)
    temp_predict_lr = clf_lr.predict_proba(test_x)[:, 1]
    res = res + rankdata(temp_predict_lr, method='ordinal')*weights[3]
    # res+=temp_predict_lr*weights[3]
    # res_temp_lr = temp_predict_lr * temp_predict_lr * weight
    del clf_lr,temp_train,temp_y
    gc.collect()

    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    clf_nn = KerasClassifier_wrapper(train_x.shape[1])
    model_path = "../input/keras_model.h5"
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=20,
            mode='max',
            verbose=100),
        ModelCheckpoint(
            model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=100)
    ]
    # fit estimator
    history = clf_nn.fit(
        train_x,
        train_y,
        epochs=1000,
        batch_size=256,
        learning_rate=0.001,
        validation_data=(val_x, val_y),
        verbose=1,
        callbacks=callbacks,
        shuffle=True
    )
    print(history.history.keys())
    import matplotlib.pyplot as plt
    # summarize history for R^2
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig_acc.savefig("model_auc.png")

    # summarize history for loss
    fig_loss = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig_loss.savefig("model_loss.png")
    temp_predict_nn = clf_nn.predict_proba(test_x)[:,1]
    res = res + rankdata(temp_predict_nn, method='ordinal')*weights[4]
    # res+=temp_predict_nn*weights[4]
    del history,clf_nn,train_x,train_y,val_x,val_y,test_x
    gc.collect()
    res= (res -min(res))/(max(res)-min(res))
    return  res

def rank_blending_predict(train_set, val_set, test_set, file, val_ratio=0.4, n_round=3):
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
    print("begin to train ")

    # val_len = int(len(val_set)*0.4)
    # use_cols = list(set(train_set.columns)-set(["user_id","label"]))
    df_val_user = pd.read_pickle("../work/val_user_17_23.pkl")
    # val_user_all = df_val_user["user_id"].unique().tolist()
    final_rank = 0
    # train_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    # val_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    # train_set.reset_index(drop=True,inplace=True)
    # val_set.reset_index(drop=True,inplace=True)
    for i in range(n_round):
        random.seed(np.random.randint(1, 1000))
        # val = val_set.iloc[-val_len:, :].sample(frac=val_ratio)
        # val = val_set.sample(frac=val_ratio)
        val_user = get_stratified_sample(df_val_user,sample_ratio=val_ratio)
        # val_user_add = list(set(val_user_all)-set(val_user))
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
        temp_predict = blending_test(train_x,train_y,val_x,val_y,test_x,weights=(0.42,0.20,0.16,0.1,0.12))
        res_temp = test_set[['user_id']]
        res_temp['prob'] = 0
        res_temp['prob'] = temp_predict
        res_temp = pd.merge(testb, res_temp, on='user_id', how='left').fillna(0)
        res_temp.to_csv('../input/' + file +str(i)+ '.txt', sep=',', index=False, header=False)
        # res_temp = get_normalized_rank(res_temp)
        # res['prob']+= res_temp['rank']
        # res['prob']+= res_temp['rank']/n_round
        # res['prob']+= res_temp['prob']/n_round
        # res['prob']+= temp_predict/n_round
        # res_temp = res_temp[["user_id","rank"]]
        final_rank = final_rank+rankdata(temp_predict, method='ordinal')
        del val, val_train,train,train_y,train_x,val_y,val_x,res_temp,temp_predict
        gc.collect()
    res["prob"] = (final_rank -min(final_rank))/(max(final_rank)-min(final_rank))
    # res["prob"] = (res["prob"] -min(res["prob"]))/(max(res["prob"])-min(res["prob"]))
    res=pd.merge(testb,res,on='user_id',how='left').fillna(0)
    res.to_csv('../work/' + file + '.txt', sep=',', index=False,header=False)
    del testb,train_set, val_set,test_set
    gc.collect()
    return res