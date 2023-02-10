import numpy as np


def dichotomy(a=-10, b=10, func=lambda x: x, times=100):
    '''
    二分法求零点
    :params a, b: 区间起终。本函数在[a,b]区间内，寻找func的零点并返回零点位置。
    :params func: 函数
    :params times: 迭代次数
    '''
    for _ in range(times+1):
        if func(a) == 0:
            return a
        elif func(b) == 0:
            return b
        elif func(a)*func(b) > 0:
            return 'error'
        x0 = (a+b)/2
        if func(x0) == 0:
            return x0
        elif func(x0)*func(a) > 0:
            a = x0
        elif func(x0)*func(b) > 0:
            b = x0
        # print('%6.6f|%6.6f' % (x0, func(x0)))
    return x0


def two_dimensional_gaussian_distribution(x, y, sigma_1: float = 1, sigma_2: float = 1, mu_1: float = 0, mu_2: float = 0, rho: float = 0):
    '''二维正态分布'''
    c1 = 1/(2*np.pi*sigma_1*sigma_2*np.sqrt(1-rho**2))
    c2 = -1/(2*(1-rho**2))
    c3 = (x-mu_1)**2/sigma_1**2 + (y-mu_2)**2/sigma_2**2 - \
        2*rho*(x-mu_1)*(y-mu_2)/(sigma_1*sigma_2)
    return c1*np.exp(c2*c3)


def gaussian_distribution(x, sigma: float = 1, mu: float = 0):
    '''正态分布'''
    return np.exp(-(x-mu)**2/2*sigma**2)/(np.sqrt(2*np.pi)*sigma)
