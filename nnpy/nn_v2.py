import pandas as pd
import numpy as np
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
        return FP / (N+0.00000001)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # P_TA prob true alerts for binary classifier
    def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true)
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)
        return TP / (P+0.00000001)
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
def nn_predict(train_set,val_set,test_set,file,minmax_scale=True,val_ratio=0.4, n_round = 3):
    import numpy as np
    from scipy.stats import rankdata
    import random
    import gc
    res=test_set[['user_id']]
    test_x = test_set.drop(['user_id',"label"], axis=1)
    res['prob'] = 0
    user_register_log = ["user_id", "register_day", "register_type", "device_type"]
    dtype_user_register = {"user_id": np.uint32, "register_day": np.uint8, "register_type": np.uint8, "device_type": np.uint8}
    testb = \
    pd.read_table('/mnt/datasets/fusai/user_register_log.txt', header=None, names=user_register_log, index_col=None,
                  dtype=dtype_user_register)[['user_id']]
    print("begin to train ")

    # val_len = int(len(val_set)*0.4)
    # use_cols = list(set(train_set.columns)-set(["user_id","label"]))
    df_val_user = pd.read_pickle("../work/val_user_8_23.pkl")
    # val_user_all = df_val_user["user_id"].unique().tolist()
    # final_rank = 0
    # train_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    # val_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    # train_set.reset_index(drop=True,inplace=True)
    # val_set.reset_index(drop=True,inplace=True)
    if minmax_scale:
        for f in test_x.columns:
            train_set[f] = (train_set[f]-train_set[f].min())/(train_set[f].max()-train_set[f].min())
            val_set[f] = (val_set[f]-val_set[f].min())/(val_set[f].max()-val_set[f].min())
            test_x[f] = (test_x[f]-test_x[f].min())/(test_x[f].max()-test_x[f].min())
    for i in range(n_round):
        random.seed(np.random.randint(1, 1000))
        # val = val_set.iloc[-val_len:, :].sample(frac=val_ratio)
        # val = val_set.sample(frac=val_ratio)
        print("get stratified sample validation user")
        val_user = get_stratified_sample(df_val_user,sample_ratio=val_ratio)
        # val_user_add = list(set(val_user_all)-set(val_user))
        val = val_set.loc[val_set["user_id"].isin(val_user)]
        # val_add = val_set.loc[val_set["user_id"].isin(val_user_add)]
        val_train = val_set.loc[~val_set["user_id"].isin(val["user_id"])]
        train = pd.concat([train_set, val_train], axis=0)
        # train = pd.concat([train_set, val_train,val_add], axis=0)
        print("the {}th round".format(i))
        print("shape of val:", val.shape)
        print("shape of train:", train.shape)
        train_y = train['label']
        train_x = train.drop(['user_id', "label"], axis=1)
        val_y = val['label']
        val_x = val.drop(['user_id', "label"], axis=1)
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
            epochs=500,
            batch_size=1024,
            validation_data=(val_x, val_y),
            verbose=1,
            callbacks=callbacks,
            shuffle=True,
            n_jobs=-1,
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
        # weight = 2*(temp_score_train*temp_score_val)/(temp_score_train+temp_score_val)
        res_temp = test_set[['user_id']]
        res_temp['prob'] = 0
        temp_predict = clf_nn.predict_proba(test_x)[:, 1]
        res_temp['prob'] = temp_predict
        res_temp = pd.merge(testb, res_temp, on='user_id', how='left').fillna(0)
        res_temp.to_csv('../input/' + file +str(i)+ '.txt', sep=',', index=False, header=False)
        # res_temp = get_normalized_rank(res_temp)
        # res['prob']+= res_temp['rank']
        # res['prob']+= res_temp['rank']/n_round
        # res['prob']+= res_temp['prob']/n_round
        res['prob']+= temp_predict/n_round
        # res_temp = res_temp[["user_id","rank"]]
        # final_rank = final_rank+rankdata(temp_predict, method='ordinal')
        del val, val_train,train,train_y,train_x,val_y,val_x,res_temp,temp_predict,clf_nn,history
        gc.collect()
    # res["prob"] = (final_rank -min(final_rank))/(max(final_rank)-min(final_rank))
    # res["prob"] = (res["prob"] -min(res["prob"]))/(max(res["prob"])-min(res["prob"]))
    res=pd.merge(testb,res,on='user_id',how='left').fillna(0)
    res.to_csv('../work/' + file + '.txt', sep=',', index=False,header=False)
    del testb,train_set, val_set,test_set
    gc.collect()
    return res