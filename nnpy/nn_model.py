import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import r2_score, log_loss, roc_auc_score
from sklearn.model_selection import KFold

# Training steps
STEPS = 500
LEARNING_RATE = 0.0001
BETA = 0.01
DROPOUT = 0.5
RANDOM_SEED = 12345
MAX_Y = 250
RESTORE = True
START = 0

# Training variables
IN_DIM = 13

# Network Parameters - Hidden layers
n_hidden_1 = 100
n_hidden_2 = 50

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)

def deep_network(inputs, keep_prob):
    # Input -> Hidden Layer
    w1 = weight_variable([IN_DIM, n_hidden_1])
    b1 = bias_variable([n_hidden_1])
    # Hidden Layer -> Hidden Layer
    w2 = weight_variable([n_hidden_1, n_hidden_2])
    b2 = bias_variable([n_hidden_2])
    # Hidden Layer -> Output
    w3 = weight_variable([n_hidden_2, 1])
    b3 = bias_variable([1])

    # 1st Hidden layer with dropout
    h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
    h1_dropout = tf.nn.dropout(h1, keep_prob)
    # 2nd Hidden layer with dropout
    h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
    h2_dropout = tf.nn.dropout(h2, keep_prob)

    # Run sigmoid on output to get 0 to 1
    out = tf.nn.sigmoid(tf.matmul(h2_dropout, w3) + b3)

    # Loss function with L2 Regularization
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

    scaled_out = tf.multiply(out, MAX_Y)  # Scale output
    return inputs, out, scaled_out, regularizers
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
def nn_model(train_set,val_set,file,best_params=None,val_ratio=0.4, n_round = 3):
    tf.set_random_seed(RANDOM_SEED)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IN_DIM])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Dropout on hidden layers
    keep_prob = tf.placeholder("float")

    # Build the graph for the deep net
    inputs, out, scaled_out, regularizers = deep_network(x, keep_prob)

    # Normal loss function (RMSE)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, scaled_out))))

    # Loss function with L2 Regularization
    loss = tf.reduce_mean(loss + BETA * regularizers)

    # Optimizer
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


    logloss = log_loss(y_, scaled_out)
    auc = roc_auc_score(y_, scaled_out)

    # Save model
    use_cols = list(set(train_set.columns)-set(["user_id","label"]))
    df_val_user = pd.read_pickle("../work/val_user_17_23.pkl")
    val_user_all = df_val_user["user_id"].unique().tolist()
    final_rank = 0
    train_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    val_set.drop_duplicates(inplace=True, subset=use_cols, keep="last")
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        #if RESTORE:
        #    print('Loading Model...')
        #    ckpt = tf.train.get_checkpoint_state('./models/neural/')
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        #else:
        sess.run(tf.global_variables_initializer())
        # val = val_set.iloc[-val_len:, :].sample(frac=val_ratio)
        # val = val_set.sample(frac=val_ratio)
        val_user = get_stratified_sample(df_val_user, sample_ratio=val_ratio)
        val_user_add = list(set(val_user_all) - set(val_user))
        val = val_set[val_set["user_id"].isin(val_user)]
        # val_add = val_set[val_set["user_id"].isin(val_user_add)]
        val_train = val_set[~val_set["user_id"].isin(val["user_id"])]
        train = pd.concat([train_set, val_train], axis=0)
        # train = pd.concat([train_set, val_train,val_add], axis=0)
        print("shape of val:", val.shape)
        print("shape of train:", train.shape)
        y_train = train['label']
        train = train.drop(['user_id', "label"], axis=1)
        val_y = val['label']
        val_x = val.drop(['user_id', "label"], axis=1)

        # Train until maximum steps reached or interrupted
        for i in range(START, STEPS):
            k_fold = KFold(n_splits=10, shuffle=True)
            #if i % 100 == 0:
            #    saver.save(sess, './models/neural/step_' + str(i) + '.cptk')

            for k, (ktrain, ktest) in enumerate(k_fold.split(train, y_train)):
                train_step.run(feed_dict={x: train[ktrain], y_: y_train[ktrain], keep_prob: DROPOUT})
                # Show test score every 10 iterations
                if i % 10 == 0:
                    # Tensorflow R2
                    #train_accuracy = accuracy.eval(feed_dict={
                    #    x: train[ktest], y_: y_train[ktest]})
                    # SkLearn metrics R2
                    train_accuracy = log_loss(y_train[ktest],
                                              sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
                    print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, train_accuracy))

        CV = []
        for i in range(n_round):
            k_fold = KFold(n_splits=10, shuffle=True)
            for k, (ktrain, ktest) in enumerate(k_fold.split(train, y_train)):
                # Tensorflow R2
                #accuracy = accuracy.eval(feed_dict={
                #    x: train[ktest], y_: y_train[ktest]})
                # SkLearn metrics R2
                auc = roc_auc_score(y_train[ktest],
                                          sess.run(scaled_out, feed_dict={x: train[ktest], keep_prob: 1.0}))
                print('Step: %d, Fold: %d, R2 Score: %g' % (i, k, auc))
                CV.append(auc)
        print('Mean R2: %g' % (np.mean(CV)))

if __name__ == '__main__':
    tf.app.run()