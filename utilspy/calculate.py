def calculate():
    M_hat = 51480
    F1_hat = 0.63088748
    precision_hat = F1_hat/(2-F1_hat)
    N_hat = M_hat*precision_hat
    print(N_hat)

    f1 = 0.8014
    M = 30000
    TP = (M+N_hat)/2*f1

    precision = TP/M
    recall = TP/N_hat
    print("True positive number {} ".format(TP))
    print("precision {}".format(precision))
    print("recall {}".format(recall))



    p = 20200/25600
    r = 20200/23722
    print("pre {}".format(p))
    print("rec {}".format(r))



    # p = 0.795
    # r = 0.845
    # print("possible submit number {}".format(N_hat*r/p))
    f1 = 2*p*r/(p+r)
    print("f1 score {}".format(f1))

    print(336/800)

if __name__ == "__main__":
    calculate()
