import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn import metrics
from statsmodels.tools import eval_measures
from scipy import stats
from warnings import filterwarnings
filterwarnings("ignore")
sns.set_style('darkgrid')
plt.rc('font', size=10)


# ###########---------------set up and plot input data-----------------######################
base_value = 10  # 设置level、trend、season项的基数
steps_day, steps_week = 1, 1
length = [steps_day*5+steps_day, steps_week*5+steps_week]  # 代表每个序列的长度，分别为周、日序列的一年及两年

weights = []
for i in range(-base_value + 1, 1):
    weights.append(0.5 ** i)  # 设置y_level项随机序列的权重呈递减指数分布，底数越小，y_level中较小值所占比例越大。
weights = np.array(weights)


# #########################################################--构造乘法周期性时间序列，模拟真实销售
y_level_actual, y_trend_actual, y_season_actual, y_noise_actual, y_input_mul_actual = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_actual[i] = np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性
    y_level_actual[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_actual[i])) + np.average(abs(y_season_actual[i]))  # 用指数权重分布随机数模拟水平项
    y_trend_actual[i] = (2 * max(y_season_actual[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_actual[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_actual[i] = 3*np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以乘法方式进入模型，则可令噪音在0附近呈学生分布。
    y_noise_actual[i][abs(y_noise_actual[i]) < max(y_noise_actual[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_actual[i] = (y_level_actual[i] + y_trend_actual[i]) * y_season_actual[i] * y_noise_actual[i]  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条真实序列中水平项的极差：{max(y_level_actual[i]) - min(y_level_actual[i])}，均值：{np.mean(y_level_actual[i])}')
    print(f'第{i}条真实序列中趋势项的极差：{max(y_trend_actual[i]) - min(y_trend_actual[i])}，均值：{np.mean(y_trend_actual[i])}')
    print(f'第{i}条真实序列中周期项的极差：{max(y_season_actual[i]) - min(y_season_actual[i])}，均值：{np.mean(y_season_actual[i])}')
    print(f'第{i}条真实序列中噪音项的极差：{max(y_noise_actual[i]) - min(y_noise_actual[i])}，均值：{np.mean(y_noise_actual[i])}')
    print(f'第{i}条真实乘法性序列最终极差：{max(y_input_mul_actual[i]) - min(y_input_mul_actual[i])}，均值：{np.mean(y_input_mul_actual[i])}', '\n')

    y_level_actual[i] = pd.Series(y_level_actual[i]).rename('y_level_actual')
    y_trend_actual[i] = pd.Series(y_trend_actual[i]).rename('y_trend_actual')
    y_season_actual[i] = pd.Series(y_season_actual[i]).rename('y_season_actual')
    y_noise_actual[i] = pd.Series(y_noise_actual[i]).rename('y_noise_actual')
    y_input_mul_actual[i] = pd.Series(y_input_mul_actual[i]).rename('y_input_mul_actual')
    # print(y_input_mul_actual[i], '\n')

# 绘制四条乘法季节性时间序列；xlim让每条折线图填充满x坐标轴
plt.figure('mul_actual_pred: day', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_mul_actual[0].plot(ax=ax1, legend=True)
y_level_actual[0].plot(ax=ax2, legend=True)
y_trend_actual[0].plot(ax=ax3, legend=True)
y_season_actual[0].plot(ax=ax4, legend=True)
y_noise_actual[0].plot(ax=ax5, legend=True)

plt.figure('mul_actual_pred: week', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_mul_actual[1].plot(ax=ax1, legend=True)
y_level_actual[1].plot(ax=ax2, legend=True)
y_trend_actual[1].plot(ax=ax3, legend=True)
y_season_actual[1].plot(ax=ax4, legend=True)
y_noise_actual[1].plot(ax=ax5, legend=True)


##########################################################--构造乘法周期性时间序列，模拟预测销售
y_level_pred, y_trend_pred, y_season_pred, y_noise_pred, y_input_mul_pred = [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length), [[]] * len(length)
for i in range(0, len(length)):
    y_season_pred[i] = 1/2 * np.sqrt(base_value) * np.sin(np.linspace(np.pi / 2, 10 * np.pi, length[i]))  # 用正弦函数模拟周期性，使预测销售的波动振幅比真实销售小
    y_level_pred[i] = np.array(random.choices(range(0, base_value), weights=weights, k=length[i])) / np.average(abs(y_season_pred[i])) + np.average(abs(y_season_pred[i]))  # 用指数权重分布随机数模拟水平项，使其相对于真实销售有所偏移
    y_trend_pred[i] = (2 * max(y_season_pred[i]) + np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)
        + (min(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)) + max(np.log(np.linspace(2, 2 ** (base_value / 8), num=length[i])) / np.log(1.1)))
        / length[i] * np.linspace(1, length[i], num=length[i])) / 10 * np.average(y_level_pred[i])  # 用对数函数与线性函数的均值模拟趋势性
    y_noise_pred[i] = np.random.standard_t(length[i]-1, length[i])  # 假定数据处于理想状态，并使噪音以乘法方式进入模型，则可令噪音在0附近呈学生分布；使其比真实销售的噪音小。
    y_noise_pred[i][abs(y_noise_pred[i]) < max(y_noise_pred[i])*0.9] = 1  # 保留随机数中的离群值，将非离群值置为1
    y_input_mul_pred[i] = (y_level_pred[i] + y_trend_pred[i]) * y_season_pred[i] * y_noise_pred[i]  # 假定周期项以乘法方式组成输入数据

    print(f'第{i}条预测序列中水平项的极差：{max(y_level_pred[i]) - min(y_level_pred[i])}，均值：{np.mean(y_level_pred[i])}')
    print(f'第{i}条预测序列中趋势项的极差：{max(y_trend_pred[i]) - min(y_trend_pred[i])}，均值：{np.mean(y_trend_pred[i])}')
    print(f'第{i}条预测序列中周期项的极差：{max(y_season_pred[i]) - min(y_season_pred[i])}，均值：{np.mean(y_season_pred[i])}')
    print(f'第{i}条预测序列中噪音项的极差：{max(y_noise_pred[i]) - min(y_noise_pred[i])}，均值：{np.mean(y_noise_pred[i])}')
    print(f'第{i}条预测乘法性序列最终极差：{max(y_input_mul_pred[i]) - min(y_input_mul_pred[i])}，均值：{np.mean(y_input_mul_pred[i])}', '\n')

    y_level_pred[i] = pd.Series(y_level_pred[i]).rename('y_level_pred')
    y_trend_pred[i] = pd.Series(y_trend_pred[i]).rename('y_trend_pred')
    y_season_pred[i] = pd.Series(y_season_pred[i]).rename('y_season_pred')
    y_noise_pred[i] = pd.Series(y_noise_pred[i]).rename('y_noise_pred')
    y_input_mul_pred[i] = pd.Series(y_input_mul_pred[i]).rename('y_input_mul_pred')
    # print(y_input_mul_pred[i], '\n')

# 绘制四条乘法季节性时间序列；xlim让每条折线图填充满x坐标轴
plt.figure('mul_actual_pred: day', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[0]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[0]-1)
y_input_mul_pred[0].plot(ax=ax1, legend=True)
y_level_pred[0].plot(ax=ax2, legend=True)
y_trend_pred[0].plot(ax=ax3, legend=True)
y_season_pred[0].plot(ax=ax4, legend=True)
y_noise_pred[0].plot(ax=ax5, legend=True)

plt.figure('mul_actual_pred: week', figsize=(5,10))
ax1 = plt.subplot(5,1,1)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax2 = plt.subplot(5,1,2)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax3 = plt.subplot(5,1,3)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax4 = plt.subplot(5,1,4)
xlim = plt.gca().set_xlim(0, length[1]-1)
ax5 = plt.subplot(5,1,5)
xlim = plt.gca().set_xlim(0, length[1]-1)
y_input_mul_pred[1].plot(ax=ax1, legend=True)
y_level_pred[1].plot(ax=ax2, legend=True)
y_trend_pred[1].plot(ax=ax3, legend=True)
y_season_pred[1].plot(ax=ax4, legend=True)
y_noise_pred[1].plot(ax=ax5, legend=True)


"""
预测结果评价函数的使用方法:
1. 评价种类、品类群、课别、企业、全体层级的结果时，将企业种类的各对预测金额及理论销售金额传入函数，而不是将各汇总层级的序列对输入函数，避免低层级的偏差信息因正负号不同而消失；
评价单品、品种层级的结果时，将企业单品的各对预测销量及理论销量传入函数。不用单品评价结果去计算种类及更高层级的精度，是因为预测系统算法设计的优化目标是企业种类预测金额，而不是直接优化单品预测销量。
2. 当同一目标有三种序列时：预测2.0、预测1.0、实际销售，应把预测2.0和实际销售、预测1.0和实际销售分别传入函数regression_accuracy，
来评价预测2.0贴近实际销售的程度、预测1.0贴近实际销售的程度。当要评价预测2.0和预测1.0的偏离程度时，应把从regression_accuracy得到的两种值，再传入函数smape得到偏离程度。
因为smape是所有指标里唯一具有无偏性的百分比指标，不是以某一种序列为目标，评价另一种序列趋近该种序列的程度；所以不应把预测2.0和预测1.0传入regression_accuracy直接得到偏离程度。
也不能将预测2.0与预测1.0拼到一起，与理论销售作为一个整体的序列对传入regression_accuracy，因为两种预测结果是由两种不同模型得到的，若拼到一起，
在后续归一化时会相互影响，进而也会影响其后的调和平均，从而不能对每种模型的预测结果得到准确的相互独立的评价结果。
3. 除了最终指标precision是最重要的评价指标外，如果更关心平时的预测结果，参考MAPE更好；如果更关心节假日期间的预测结果，参考RMSE(eval_measures.rmse)更好。
4. 若要评价某个具体的企业种类的预测结果，不能将这个序列对输入regression_accuracy，可以输入分项函数中的任意一个；
若要评价某个种类的预测结果，则将各个企业种类的预测序列和真实序列组成的序列对传入regression_accuracy，对返回结果按序列求平均，得到该种类的预测评价结果；
此时序列对中的y_true根据企业的不同而不同；而不是将该种类的汇总序列对传入regression_accuracy。若要对某个具体的企业单品、某个单品的预测结果进行评价，与4中方法类似。
5. 若要评价某个企业品类群的预测结果，则将该企业品类群下所有企业种类序列对传入regression_accuracy，对返回结果按序列求平均，此时序列对中的y_true是不同的，因为是不同的企业种类的理论销售；
而不是将该品类群的汇总序列对传入regression_accuracy。若要对某个企业课别、某个企业、全体的预测结果进行评价，与5中方法类似；
若要对某个企业品种、某个品种的预测结果进行评价，也与5中方法类似，只是传入regression_accuracy的是由企业单品构成的序列对。
6. 若要评价（优选）不同算法模型或不同中间计算结果对同一企业种类的预测结果，则传入regression_accuracy的序列对中y_true是相同的，因为是同一企业种类的理论销售。
7. 综上，可完成对任一层级、每个层级下任一子集的准确评价。
"""


# y_true, y_pred无限制条件
def emlae(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等
    :return: EMLAE，将残差离群值压缩后的一次绝对性指标，对残差的离群值不如MAE敏感
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    emlae = np.exp(sum(np.log(abs(y_pred - y_true) + 1)) / n) - 1
    return emlae

# y_true != 0
def mape(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等，且不能为0
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等
    :return: MAPE，零次的相对性指标，小数，代表预测值偏离真实值的平均程度；若>1，表示预测值偏离真实值的平均程度超过真实值的1倍，若<1，表示预测值偏离真实值的平均程度小于真实值的1倍
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = sum(abs((y_pred - y_true) / y_true)) / n
    return mape

# y_true + y_pred != 0
def smape(y_true, y_pred):
    """
    :param y_true: 一条真实值序列，array,series,list,tuple均可，长度要与预测值长度相等，与对应预测值不能同时为0
    :param y_pred: 一条预测值序列，array,series,list,tuple均可，长度要与真实值长度相等，与对应真实值不能同时为0
    :return: SMAPE，零次的相对性指标，小数，代表预测值偏离真实值的平均程度；不以某类特定的量为基准分母，而是将两类值均作为分母，第一具有对称性，
    第二不会因某类量数值的大小而影响评价结果，所以受离群值影响也比MAPE小；若>1，表示预测值偏离真实值的平均程度超过真实值与预测值均值的1倍，若<1，表示预测值偏离真实值的平均程度小于真实值与预测值均值的1倍。
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    smape = sum(abs(2 * (y_pred - y_true) / (y_pred + y_true))) / n
    return smape

def male(y_true, y_pred):
    """
    param：
        Y:原始序列，array,series,list,tuple均可
        y:拟合序列，array,series,list,tuple均可
    return：
        对数MAE值
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[y_true < 0] = 0
    y_pred[y_pred == -1] = -0.99
    male = sum(abs(np.log(abs(y_true+1)) - np.log(abs(y_pred+1)))) / len(y_true)
    return male


def regression_accuracy(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_pred中的预测序列按顺序一一对应
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_true中的真实序列按顺序一一对应
    :return: 精度指标，按顺序分别是：最终精度指标，MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    if (len(y_true) != len(y_pred)) or (len(y_true) < 2):
        raise Exception('y_true与y_pred中序列条数必须相等且≥2')

    MAPE, SMAPE, RMSPE, MTD_p2 = [], [], [], []  # 零次的相对性指标
    EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1 = [], [], [], [], [], []  # 一次的绝对性指标
    MSE, MSLE = [], []  # 二次的绝对性指标

    y_true_trun, y_pred_trun = [], []
    for i in range(len(y_true)):
        # 为了统一下列12个精度指标的条件，在y_true和y_pred的对应序列中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效的精度值
        judge = (y_true[i] > 0.01) & (y_pred[i] > 0.01)
        if sum(judge):
            y_true_trun.append(y_true[i][judge])
            y_pred_trun.append(y_pred[i][judge])
            # 第一组，零次的相对性指标：
            MAPE.append(mape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true != 0; no bias
            SMAPE.append(smape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
            RMSPE.append(eval_measures.rmspe(np.array(y_true_trun[i]), np.array(y_pred_trun[i])) / 10)  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
            MTD_p2.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=2)) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred[i] / y_true[i] > 1, nevertheless, more susceptible to outliers than MAPE when y_pred[i] / y_true[i] < 1
            # 第二组，一次的绝对性指标：
            EMLAE.append(emlae(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
            MALE.append(male(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件;
            MAE.append(metrics.mean_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
            RMSE.append(eval_measures.rmse(np.array(y_true_trun[i]), np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
            MedAE.append(metrics.median_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
            MTD_p1.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=1))  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.
            # 第三组，二次的绝对性指标：
            MSE.append(metrics.mean_squared_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
            MSLE.append(metrics.mean_squared_log_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm
        else: continue

    print('判断前的真实（及预测）序列条数:', len(y_true), '  判断后的真实（及预测）序列条数:', len(y_true_trun), '\n')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2, '\n')
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1, '\n')
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE, '\n')

    # 将各序列对的若干精度指标整合成各序列对的最终单一评价指标；序列对的数目必须≥2，否则归一化不能起到抹平不同指标不同数量级的作用；
    # 将各精度指标按序列对的维度进行归一化
    MAPE_1 = np.array(MAPE) / sum(np.array(MAPE))
    SMAPE_1 = np.array(SMAPE) / sum(np.array(SMAPE))
    RMSPE_1 = np.array(RMSPE) / sum(np.array(RMSPE))
    MTD_p2_1 = np.array(MTD_p2) / sum(np.array(MTD_p2))

    EMLAE_1 = np.array(EMLAE) / sum(np.array(EMLAE))
    MALE_1 = np.array(MALE) / sum(np.array(MALE))
    MAE_1 = np.array(MAE) / sum(np.array(MAE))
    RMSE_1 = np.array(RMSE) / sum(np.array(RMSE))
    MedAE_1 = np.array(MedAE) / sum(np.array(MedAE))
    MTD_p1_1 = np.array(MTD_p1) / sum(np.array(MTD_p1))

    MSE_1 = np.array(MSE) / sum(np.array(MSE))
    MSLE_1 = np.array(MSLE) / sum(np.array(MSLE))

    # 用简单调和平均计算各序列对经归一化后的各个精度指标，形成各序列对的最终单一精度指标；调和平均受输入样本点离群值的影响最小，相比于其他平均值而言；
    precision = []
    for i in range(len(y_true_trun)):
        precision.append(stats.hmean([MAPE_1[i], MAPE_1[i], MAPE_1[i], SMAPE_1[i], SMAPE_1[i], RMSPE_1[i], RMSPE_1[i], MTD_p2_1[i],
                                      EMLAE_1[i], MALE_1[i], MAE_1[i], RMSE_1[i], RMSE_1[i], RMSE_1[i], MedAE_1[i], MTD_p1_1[i],
                                      MSE_1[i], MSLE_1[i]]))
    print('各序列对的最终精度：', '\n', np.array(precision), '\n')

    return precision, MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE  # 注意返回的各分量精度指标是未归一化前的数值，而最终precision是由各分量精度指标归一化后的数值算出的

results = regression_accuracy(y_true=y_input_mul_actual, y_pred=y_input_mul_pred)
results = pd.DataFrame(results, columns=['couple one', 'couple two'], index=['precision',
                                                                   'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2',
                                                                   'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
                                                                   'MSE', 'MSLE'])
print('指标数：', len(results))
print(results, '\n')


def regression_evaluation(y_true, y_pred):
    """
    :param y_true: 若干条真实序列组成的一个二维list或array或series，其中的每条真实序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_pred中的预测序列按顺序一一对应
    :param y_pred: 若干条预测序列组成的一个二维list或array或series，其中的每条预测序列必须是带索引的series，为了能对>0的数值的索引取交集；并与y_true中的真实序列按顺序一一对应
    :return: 精度指标，按顺序分别是：最终精度指标，MAPE, SMAPE, RMSPE, MTD_p2, EMLAE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE

    测试几种常用的评价序列对的精度指标
    原则：1.带平方项的指标会放大在正负1之外的残差的影响，而压缩在正负1之内的残差的影响，由于各指标越接近零越好，则会惩罚正负1之外的残差，偏离正负1越远，越受到惩罚；而奖励正负1之内的残差。
    2.作对数变换的指标会降低离群值的影响，相对于不带对数项的指标，会惩罚非离群值。因为在(1,+∞)区间内，y=lnx的一阶导数比y=x的一阶导数小，则y=lnx比y=x递增慢。
    3.ln(1/x)+len(x)=0，即对一个数（如x）取对数，与其倒数（1/x）取对数，互为相反数；当x>0，ln(1/x)+x-1≥0，lnx+(1/x)-1≥0，可由求导证明；在(0,4]的区间内，lnx的增长速度快于x**(1/2)，在[4,+∞)区间内，lnx的增长速度慢于x**(1/2)，可由求导证明。
    """

    if (len(y_true) != len(y_pred)) or (len(y_true) < 2):
        raise Exception('y_true与y_pred中序列条数必须相等且≥2')

    MAPE, SMAPE, RMSPE, MTD_p2, VAR = [], [], [], [], []  # 零次的相对性指标
    EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1 = [], [], [], [], [], []  # 一次的绝对性指标
    MSE, MSLE = [], []  # 二次的绝对性指标

    y_true_trun, y_pred_trun = [], []
    for i in range(len(y_true)):
        # 为了统一下列12个精度指标的条件，在y_true和y_pred的对应序列中，取大于0的对应点，即排除≤0的对应点；但不应取>0，可以取>0.01，否则若序列中存在大于0但非常接近0的数做分母，可能产生很大的值，不利于得到有效的精度值
        judge = (y_true[i] > 0.01) & (y_pred[i] > 0.01)
        if sum(judge):
            y_true_trun.append(y_true[i][judge])
            y_pred_trun.append(y_pred[i][judge])
            # 第一组，零次的相对性指标：
            MAPE.append(mape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true != 0; no bias
            SMAPE.append(smape(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true + y_pred != 0; symmetric MAPE, no bias and more general, less susceptible to outliers than MAPE.
            RMSPE.append(eval_measures.rmspe(np.array(y_true_trun[i]), np.array(y_pred_trun[i])) / 10)  # y_true != 0; susceptible to outliers of deviation ratio, if more, RMSPE will be larger than MAPE.
            MTD_p2.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=2)) # y_pred > 0, y_true > 0; less susceptible to outliers than MAPE when y_pred[i] / y_true[i] > 1, nevertheless, more susceptible to outliers than MAPE when y_pred[i] / y_true[i] < 1
            VAR.append(metrics.explained_variance_score(y_true=y_true_trun[i], y_pred=y_pred_trun[i]))  # y_true, y_pred无限制条件；但explained_variance_score为极大化目标函数，值域为(-∞, 1]，越趋近1越好；与其余的极小化目标函数相反，它们的因变量是越小越好。
            # 第二组，一次的绝对性指标：
            EMLAE.append(emlae(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件; less susceptible to outliers of error than MAE, so this will penalize small deviation and award large deviation relative to MAE.
            MALE.append(male(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件;
            MAE.append(metrics.mean_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric has no penalty, no bias
            RMSE.append(eval_measures.rmse(np.array(y_true_trun[i]), np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件；susceptible to outliers of error than MAE, so this will penalize large deviation and award small deviation relative to MAE.
            MedAE.append(metrics.median_absolute_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； if len(y) is slightly large; won't be affected by outliers completely
            MTD_p1.append(metrics.mean_tweedie_deviance(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i]), power=1))  # y_pred > 0, y_true ≥ 0; The higher `p` the less weight is given to extreme deviations between true and predicted targets.
            # 第三组，二次的绝对性指标：
            MSE.append(metrics.mean_squared_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true, y_pred无限制条件； this metric penalizes a large residual greater than a small residual because of square
            MSLE.append(metrics.mean_squared_log_error(y_true=np.array(y_true_trun[i]), y_pred=np.array(y_pred_trun[i])))  # y_true≥0, y_pred≥0； this metric penalizes an under-predicted estimate greater than an over-predicted estimate because of logarithm
        else: continue

    print('判断前的真实（及预测）序列条数:', len(y_true), '  判断后的真实（及预测）序列条数:', len(y_true_trun), '\n')
    print('第一组，零次的相对性指标：', '\n', 'MAPE:', MAPE, '\n', 'SMAPE:', SMAPE, '\n', 'RMSPE:', RMSPE, '\n', 'MTD_p2:', MTD_p2, '\n', 'VAR:', VAR, '\n')
    print('第二组，一次的绝对性指标：', '\n', 'EMLAE:', EMLAE, '\n', 'MALE:', MALE, '\n', 'MAE:', MAE, '\n', 'RMSE:', RMSE, '\n', 'MedAE:', MedAE, '\n', 'MTD_p1:', MTD_p1, '\n')
    print('第三组，二次的绝对性指标：', '\n', 'MSE:', MSE, '\n', 'MSLE:', MSLE, '\n')

    # 将各序列对的若干精度指标整合成各序列对的最终单一评价指标；序列对的数目必须≥2，否则归一化不能起到抹平不同指标不同数量级的作用；
    # 将各精度指标按序列对的维度进行归一化
    MAPE_1 = np.array(MAPE) / sum(np.array(MAPE))
    SMAPE_1 = np.array(SMAPE) / sum(np.array(SMAPE))
    RMSPE_1 = np.array(RMSPE) / sum(np.array(RMSPE))
    MTD_p2_1 = np.array(MTD_p2) / sum(np.array(MTD_p2))
    VAR_1 = (-np.array(VAR)+1 + np.mean(-np.array(VAR)+1)) / sum(-np.array(VAR)+1 + np.mean(-np.array(VAR)+1))  # 因为VAR的取值范围是(-∞, 1]，越趋近1越好，是极大化目标函数，与其他指标相反；所以需要对其做数值变换，使其变为极小化目标函数，并加上np.mean，使其更适合作归一化。

    EMLAE_1 = np.array(EMLAE) / sum(np.array(EMLAE))
    MALE_1 = np.array(MALE) / sum(np.array(MALE))
    MAE_1 = np.array(MAE) / sum(np.array(MAE))
    RMSE_1 = np.array(RMSE) / sum(np.array(RMSE))
    MedAE_1 = np.array(MedAE) / sum(np.array(MedAE))
    MTD_p1_1 = np.array(MTD_p1) / sum(np.array(MTD_p1))

    MSE_1 = np.array(MSE) / sum(np.array(MSE))
    MSLE_1 = np.array(MSLE) / sum(np.array(MSLE))

    # 用简单调和平均计算各序列对经归一化后的各个精度指标，形成各序列对的最终单一精度指标；调和平均受输入样本点离群值的影响最小，相比于其他平均值而言；
    precision = []
    for i in range(len(y_true_trun)):
        precision.append(stats.hmean([MAPE_1[i], MAPE_1[i], MAPE_1[i], SMAPE_1[i], SMAPE_1[i], RMSPE_1[i], RMSPE_1[i], MTD_p2_1[i], VAR_1[i],
                                      EMLAE_1[i], MALE_1[i], MAE_1[i], RMSE_1[i], RMSE_1[i], RMSE_1[i], MedAE_1[i], MTD_p1_1[i],
                                      MSE_1[i], MSLE_1[i]]))
    print('各序列对的最终精度：', '\n', np.array(precision), '\n')

    return precision, MAPE, SMAPE, RMSPE, MTD_p2, VAR, EMLAE, MALE, MAE, RMSE, MedAE, MTD_p1, MSE, MSLE  # 注意返回的各分量精度指标是未归一化前的数值，而最终precision是由各分量精度指标归一化后的数值算出的

results = regression_evaluation(y_true=y_input_mul_actual, y_pred=y_input_mul_pred)
results = pd.DataFrame(results, columns=['couple one', 'couple two'], index=['precision',
                                                                   'MAPE', 'SMAPE', 'RMSPE', 'MTD_p2', 'VAR',
                                                                   'EMLAE', 'MALE', 'MAE', 'RMSE', 'MedAE', 'MTD_p1',
                                                                   'MSE', 'MSLE'])
print('指标数：', len(results))
print(results)
