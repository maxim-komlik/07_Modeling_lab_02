import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import inspect


class Generator:
    p = None
    k = None
    b = None

    def __init__(self, k, b, p=13):
        self.k = int(k)
        self.b = int(b)
        self.p = int(p)

    def next(self, a=None, b=None):
        if a is None:
            a = 0
        else:
            a = int(a)
        if b is None:
            b = self.b
        else:
            b = int(b)
        self.p = a + (self.p * self.k) % b
        return self.p

    def uniform(self, a, b):
        return a + (b - a) * self.next()

    def normal(self, m, sigma, n=6):
        n = int(n)
        _sum = sum([self.next() for _ in range(n)])
        return m + sigma * int(np.sqrt((12 + n - 1) / n)) * (_sum - int(n / 2))

    def exponential(self, alpha):
        return -(1 / alpha) * np.log(self.next())

    def gamma(self, alpha, nu):
        return -(1 / alpha) * np.log(reduce(lambda x, y: x * y, [self.next() for _ in range(nu)], 1))

    def triangle(self, a, b, inv=False):
        _func = np.amax if inv else np.amin
        return a + (b - a) * _func([self.next(), self.next()])

    def simpson(self, a, b):
        args = [self.next(a / 2, b / 2) for _ in range(2)]
        return sum(args)


def _mean(lst):
    result = np.float64(0.0)
    for item in lst:
        result += item / len(lst)
    return result


def _std(lst, m=None):
    if m is None:
        m = _mean(lst)
    rate = np.float64(1.0) / (len(lst) - 1)
    result = np.float64(0.0)
    for item in lst:
        result += (item ** 2 - m ** 2) * rate
    return np.sqrt(result)


def _hist(x, n_bins=20):
    n_bins = int(n_bins)
    if type(x) != np.ndarray:
        x = np.array(x)
    x_min = np.amin(x)
    x_range = np.amax(x) - x_min
    bin_step = float(x_range) / n_bins
    bins = [bin_step * i for i in range(n_bins + 1)]
    result = [0] * n_bins
    for item in x:
        result[int((item - x_min) / bin_step) - 1] += 1
    return bins, result


def f1(lst):
    max_seq_len = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j]:
                if j - i > max_seq_len:
                    max_seq_len = j - i
                break
        break
        # if i >= len(lst) - max_seq_len:
        #    break
    return max_seq_len


def f2(lst):
    min_seq_len = len(lst)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j]:
                if j - i < min_seq_len:
                    min_seq_len = j - i
                break
        break
    return min_seq_len


if __name__ == '__main__':
    # 13, 73009, 63949
    params = [13, 73009, 63949]
    _labels = [f"R0 parameter: ", f"a parameter: ", f"b parameter: "]
    for i in range(len(params)):
        temp = int(input(_labels[i]))
        params[i] = temp if temp != 0 else params[i]

    gen = Generator(params[0], params[1], params[2])

    _call_args = dict()
    _options = ["uniform", "normal", "exponential", "gamma", "triangle", "simpson"]
    _callables = [gen.uniform, gen.normal, gen.exponential, gen.gamma, gen.triangle, gen.simpson]
    for i in range(len(_options)):
        print(f"{i + 1}. {_options[i]}")

    _callable_index = int(input()) - 1
    if _callable_index < 0:
        _callable_index = 0

    _callable_arg_names = inspect.getfullargspec(_callables[_callable_index])[0]
    for i in range(1, len(_callable_arg_names)):
        _call_args[_callable_arg_names[i]] = float(input(_callable_arg_names[i]))

    print(_call_args)

    def _call():
        return _callables[_callable_index](**_call_args)

    history = []
    for i in range(100000):
        history.append(_call())
    h_mean = _mean(history)
    h_std = _std(history, h_mean)
    h_max_seq_len = f1(history)
    h_min_seq_len = f2(history)
    bins, hist = _hist(history)
    print(f"Expectation: {h_mean:.5f} (effective mean)")
    print(f"Standard deviation: {h_std:.5f} (corrected)")
    print(f"Maximum unique sequence length: {h_max_seq_len}")
    print(f"Minimum unique sequence length: {h_min_seq_len}")
    plt.figure().add_subplot()
    plt.gca().hist(bins[:-1], bins, weights=hist)
    plt.show()
