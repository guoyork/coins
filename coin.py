import numpy as np

prior = 0.5
N = 3
common_num = 1
private_num = 1


def sample_coins(private_num, common_num, p):
    common_coins = np.random.binomial(1, p, size=common_num)
    private_coins = []
    for i in range(len(private_num)):
        private_coins.append(np.random.binomial(1, p, size=private_num[i]))
    return common_coins, private_coins


def report(common_coins, private_coins):
    return np.average(np.append(common_coins, private_coins))


def prediction(common_coins, private_coins, common_num, private_nums, index):
    predict = []
    common_p = np.average(common_coins)
    post_p = report(common_coins, private_coins)

    for i in range(len(private_nums)):
        if i != index:
            alpha = common_num/(common_num+private_nums[i])
            predict.append(alpha*common_p+(1-alpha)*post_p)
    return np.average(predict)


def simple_average(reports):
    return np.average(reports)


def weighted_average(reports, predictions):
    sum_reports = np.sum(reports)
    #weight = 1/np.maximum(1e-7, np.abs([predictions[0]-reports[1], predictions[1]-reports[0]]))
    weight = 1/np.maximum(1e-7, np.abs(predictions-(sum_reports-reports)/(len(reports)-1)))
    if np.sum(weight) != 0:
        weight = weight/np.sum(weight)
        return np.dot(weight, reports)
    return np.average(reports)


def my_opt(reports, predictions):
    p1 = np.average(reports)
    if reports[0] == reports[1]:
        return np.average(reports)
    alpha = np.abs(reports[1]-predictions[1])/np.abs(reports[1]-p1)
    beta = np.abs(reports[0]-predictions[0])/np.abs(reports[0]-p1)
    res = reports[0]*beta+reports[1]*alpha-p1*alpha*beta
    res /= alpha+beta-alpha*beta
    return res


def dis(a, b):
    return (a-b)**2


def exp_main(N, p):
    print("number of agents: ", N)
    print("prior: ", p)
    simple_ave = []
    weight_ave = []
    temp_ave = []
    for i in range(10000):
        private_nums = np.ones(N, dtype=int)*private_num
        total_num = common_num+np.sum(private_nums)
        common_coins, private_coins = sample_coins(private_nums, common_num, p)
        reports = []
        predictions = []
        for i in range(N):
            reports.append(report(common_coins, private_coins[i]))
            predictions.append(prediction(common_coins, private_coins[i], common_num, private_nums, i))

        reports = np.array(reports)
        predictions = np.array(predictions)
        opt = np.sum(common_coins)
        for i in range(N):
            opt += np.sum(private_coins[i])
        opt /= total_num
        simple_ave.append(dis(opt, simple_average(reports)))
        weight_ave.append(dis(opt, weighted_average(reports, predictions)))
        #temp_ave.append(dis(opt, my_opt(reports, predictions)))
        #print("average ", base)
        #print("opt ", opt)
    print(np.average(simple_ave))
    print(np.average(weight_ave))


def exp_different_N():
    for N in range(2, 20):
        exp_main(N=N, p=0.5)


def exp_different_p():
    for p in range(20):
        exp_main(N=2, p=p/20)


if __name__ == "__main__":
    exp_different_N()
