import csv
import datetime
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import keras.backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, GaussianNoise, Conv1D, MaxPooling1D, Flatten, Embedding
from keras.preprocessing.image import ImageDataGenerator
from catboost import CatBoostClassifier
from sklearn.decomposition import  PCA
from sklearn.metrics import classification_report, f1_score

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def predict(clf2, test_set, param, kpca):
    uid = pd.DataFrame()
    # test_set = processing(trainSpan=(1, 30), label=False)
    uid["user_id"] = test_set["user_id"]
    test_set = test_set.drop(labels=["user_id"], axis=1)
    test_set = kpca.transform(test_set.values)
    print("begin to make predictions")
    # res = clf2.predict_proba(test_set.values)
    res = np.reshape(clf2.predict_proba(test_set),-1)
    uid["proba1"] = pd.Series(res)
    uid["score"] = uid.groupby(by=["user_id"])["proba1"].transform(lambda x: sum(x) / float(len(x)))
    uid.drop_duplicates(subset=["user_id"], inplace=True)
    uid.sort_values(by=["score"], axis=0, ascending=False, inplace=True)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    uid_file = "../result/uid/uid_dnn_" + param + "_" + str_time + ".csv"
    uid.to_csv(uid_file, header=True, index=False)
    # active_users = uid.loc[uid["score"]>0.5]["user_id"].unique().tolist()
    active_users = uid["user_id"][:24500].unique().tolist()
    # print(len(active_users))
    print(uid["score"].tolist()[24500])
    # print(active_users)
    str_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    submission_file = "../result/622/submission_dnn_" + param + "_" + str_time + ".csv"
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
    val_file_name = file_name + "ld1-22.csv"
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
    pca_std = np.std(train_set)
    # # print("eigenvalues of the centered kernel matrix {}".format(kpca.lambdas_))
    NCOMPONENTS = kpca.n_components_
    print("number of components {}".format(kpca.n_components_))
    print("noise variance {}".format(kpca.noise_variance_))
    print("the explained variance {}".format(kpca.explained_variance_))
    print("the explained variance ratio {}".format(kpca.explained_variance_ratio_))

    print("begin to make prediction with plain features and without tuning parameters")

    # scoring = {'f1': "f1"}
    # clf1 = GridSearchCV(LGBMClassifier(),
    #                   param_grid={"n_estimators":[200,400,600],"num_leaves": [4,5,6,8],"boosting_type":["dart"]},
    #                   scoring=scoring, cv=4, refit='f1',n_jobs=-1,verbose=1)

    for layers in [3]:
        for units in [128]:
            print({"layers": layers, "neurals": units})
            model = Sequential()
            # model.add(Dense(units, input_dim=NCOMPONENTS, activation='relu'))
            # model.add(Embedding(units,32, input_lenth=NCOMPONENTS))
            # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            # model.add(Flatten())
            # model.add(Dense(250, activation='relu'))
            # model.add(Dense(1, activation='sigmoid'))
            model.add(Dense(units, input_dim=NCOMPONENTS, activation='relu'))
            model.add(GaussianNoise(pca_std))
            for i in range(layers):
                model.add(Dense(units, activation='relu'))
                model.add(GaussianNoise(pca_std))
                model.add(Dropout(0.1))
            model.add(Dense(1, activation='sigmoid'))
            print(model.summary())
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
            early_stopping = EarlyStopping(monitor="val_loss",patience=16)
            model.fit(train_set, train_label, epochs=300, batch_size=256, validation_split=0.15, verbose=2,callbacks=[early_stopping])

            print("begin to make classification report for the validation dataset")
            # yhat = clf1.predict(val_set.values)
            # yhat = clf1.predict(val_set.values)
            yhat = np.reshape(model.predict_classes(val_set),-1)

            print(classification_report(y_pred=yhat, y_true=val_label.values, digits=4))

            print("begin to make classification report for the training dataset")
            # yhat = clf1.predict(train_set.values)
            yhat = np.reshape(model.predict_classes(train_set),-1)
            print(classification_report(y_pred=yhat, y_true=train_label.values, digits=4))

            print("load the test dataset")
            test_file_name = file_name.replace("training", "testing") + "ld1-30.csv"
            test_set = pd.read_csv(test_file_name, header=0, index_col=None, usecols=keep_feature + ["user_id"])
            # test_set = pd.read_csv("data/testing_rld1-30.csv",header=0,index_col=None)
            for fea in keep_feature:
                test_set[fea] = (test_set[fea] - test_set[fea].min()) / (test_set[fea].max() - test_set[fea].min())
                # test_set[fea] = (test_set[fea]-test_set[fea].mean())/(test_set[fea].std())

            print("begin to make prediction")
            param = list(file_name)[-1] + str(scheme_num) + "_" + str(layers) + "_" + str(units)
            print(param)
            # predict(clf1,test_set,param)
            predict(model, test_set, param, kpca)

if __name__ == "__main__":
    file_name1 = "../data/data_v3/training_e"
    file_name2 = "../data/data_v4/training_r"
    for scheme in [3]:
        for file in ["../data/data_v4/training_r"]:
            run(scheme_num=scheme,file_name=file)