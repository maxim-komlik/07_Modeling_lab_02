import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import inspect
import queue


class Generator:
    p = None
    k = None
    b = None

    def __init__(self, k: int, b: int, p: int = 13):
        self.k = k
        self.b = b
        self.p = p

    def next(self, a: int = None, b: int = None):
        if a is None:
            a = 0
        else:
            a = int(a)
        if b is None:
            b = self.b
        else:
            b = int(b)
        self.p = a + (self.p * self.k) % (b - a)
        return self.p/(b - a)

    def uniform(self, a: float, b: float):
        return a + (b - a) * self.next()

    def normal(self, m: float, sigma: float, n: int = 6):
        n = int(n)
        _sum = sum([self.next() for _ in range(n)])
        return m + sigma * int(np.sqrt((12 + n - 1) / n)) * (_sum - int(n / 2))

    def exponential(self, alpha: float):
        return -(1 / alpha) * np.log(self.next())

    def gamma(self, alpha: float, nu: int):
        return -(1 / alpha) * np.log(reduce(lambda x, y: x * y, [self.next() for _ in range(nu)], 1))

    def triangle(self, a: float, b: float, inv: bool = False):
        _func = np.amax if inv else np.amin
        return a + (b - a) * _func([self.next(), self.next()])

    def simpson(self, a: float, b: float):
        args = [self.next(a / 2, b / 2) for _ in range(2)]
        return sum(args)


class SequenceAnalyzer:
    @staticmethod
    def mean(lst):
        result = np.float64(0.0)
        for i in range(len(lst)):
            result = result * (i / (i+1)) + (lst[i] / (i + 1))
        return result

    @staticmethod
    def std(lst, m: float = None):
        if m is None:
            m = SequenceAnalyzer.mean(lst)
        rate = np.float64(1.0) / (len(lst) - 1)
        result = np.float64(0.0)
        for item in lst:
            result += (item**2 - m**2) * rate
        return np.sqrt(result), result

    @staticmethod
    def hist(x, n_bins: int = 20):
        if type(x) != np.ndarray:
            x = np.array(x)
        x_min = np.amin(x)
        x_range = np.amax(x) - x_min
        bin_step = float(x_range) / n_bins
        bins = [x_min + bin_step * i for i in range(n_bins + 1)]
        if bin_step > 0:
            result = [0] * n_bins
            x = (x - x_min) / bin_step - 1
            i_x = x.astype(int)
            for item in i_x.flat:
                result[item] += 1
        else:
            raise ZeroDivisionError
        return bins, result

    @staticmethod
    def lehmer_analysis(lst):
        # 1 702551 958821
        # 1 125566 276128
        # 8628 8632

        period = len(lst)
        max_unique = -1
        i = len(lst) - 1
        while i > 0:
            j = i - 1
            low_bound = max(0, i - period)
            while j >= low_bound:
                if lst[i] == lst[j]:
                    if i - j < period:
                        period = i - j
                    break
                j -= 1
            else:
                i -= 1
                continue
            break

        if period < len(lst):
            i = 0
            while lst[i] != lst[i+period]:
                i += 1
            else:
                max_unique = i + period
                if i > 0:
                    max_unique -= 1
        return period, max_unique

    @staticmethod
    def general_analysis(lst):
        periodical_ranges = []

        max_seq_len = -1
        min_period = len(lst)
        i = len(lst) - 1
        while i > 0:
            j = i - 1
            low_bound = max(0, i - min_period)
            while j >= low_bound:
                if lst[i] == lst[j]:
                    if i - j < min_period:
                        min_period = i - j
                    k = i
                    while True:
                        i -= 1
                        j -= 1
                        if not (j > 0 and lst[i] == lst[j]):
                            break
                    periodical_ranges.append(tuple((k, i-j, k-i)))
                    if max_seq_len < k - i:
                        max_seq_len = k - i
                    break
                j -= 1
            else:
                i -= 1

        def range_contains(x, p):
            return x[0] <= p < (x[0] + x[1])

        # ... range[0] + range[1]
        # range[0] + range[2] ...
        # ... range0[0] ... range1[0] + range1[1] if range0[2] < range0[1]
        sorted_ranges = []
        for item in periodical_ranges:
            sorted_ranges.append(tuple((item[0] - item[1] - item[2], item[1], item[2])))
        sorted_ranges = sorted(sorted_ranges, key=lambda item: item[0])

        unique_seq_lens = []
        start_points = queue.Queue(len(sorted_ranges))
        start_points.put(0)
        start_range = 0
        try:
            while True:
                point = start_points.get(False)
                while point > sorted_ranges[start_range][0] and \
                        not range_contains(sorted_ranges[start_range], point):
                    start_range += 1
                    if start_range >= len(sorted_ranges):
                        break
                else:
                    current = sorted_ranges[start_range]
                    if sorted_ranges[start_range][1] > sorted_ranges[start_range][2]:
                        high_bound = current[0] + current[1]
                        for i in range(start_range + 1, len(sorted_ranges)):
                            if range_contains(current, sorted_ranges[i][0]):
                                if range_contains(current, sorted_ranges[i][0] + sorted_ranges[i][1]):
                                    if sorted_ranges[i][0] + sorted_ranges[i][1] < high_bound:
                                        high_bound = sorted_ranges[i][0] + sorted_ranges[i][1]
                                    start_points.put(sorted_ranges[i][0] + sorted_ranges[i][2])
                            else:
                                break
                        unique_seq_lens.append(high_bound - point)
                        if high_bound == current[0] + current[1]:
                            start_points.put(high_bound)
                    else:
                        unique_seq_lens.append(current[0] + current[1] - point)
                        start_points.put(current[0] + current[2])
                    continue
                unique_seq_lens.append(len(lst) - point)
                break
        except queue.Empty:
            pass

        max_unique = max(unique_seq_lens)
        return min_period, max_seq_len, max_unique

    @staticmethod
    def geometry_test(x):
        counter = 0
        for i in range(0, len(x)-1, 2):
            if (x[i]**2 + x[i+1]**2) < 1:
                counter += 1
        return (2 * counter) / len(x)


def lab():
    # 13, 73009, 63949
    # params = [13, 73009, 63949]
    params = [131, 102191, 203563]
    _labels = [f"R0 parameter: ", f"k parameter: ", f"b parameter: "]
    print(f"(R * k) % b")
    for i in range(len(params)):
        try:
            temp = int(input(_labels[i]))
            params[i] = temp if temp != 0 else params[i]
        except ValueError:
            continue

    gen = Generator(params[0], params[1], params[2])

    _call_args = dict()
    _options = ["uniform", "normal", "exponential", "gamma", "triangle", "simpson"]
    _callables = [gen.uniform, gen.normal, gen.exponential, gen.gamma, gen.triangle, gen.simpson]
    for i in range(len(_options)):
        print(f"{i + 1}. {_options[i]}")

    _callable_index = int(input()) - 1
    if _callable_index < 0:
        _callable_index = 0

    _option_labels = [
        f"1/(b-a)",
        f"(1/(sigma * sqrt(2*pi)) * e^(-(x-m)/(2*sigma^2))",
        f"alpha * e^(-alpha * x)",
        f"((alpha^nu)/((nu - 1)!)) * x^(nu-1) * e^(-alpha * x)",
        f"2*(x-a)/((b-a)^2)",
        f"4*[(x-a)|(b-x)]/((b-a)^2)",
    ]
    print(_option_labels[_callable_index])
    _callable_arg_names = inspect.getfullargspec(_callables[_callable_index])[0]
    _callable_arg_types = getattr(_callables[_callable_index], "__annotations__")
    for i in range(1, len(_callable_arg_names)):
        _call_args[_callable_arg_names[i]] = _callable_arg_types[_callable_arg_names[i]](input(_callable_arg_names[i]))

    def _call():
        return _callables[_callable_index](**_call_args)

    history = []
    for i in range(100000):
        history.append(_call())
    h_mean = SequenceAnalyzer.mean(history)
    h_std, h_var = SequenceAnalyzer.std(history, h_mean)

    # h_period, h_unique_len = SequenceAnalyzer.lehmer_analysis(history)
    # print(f"Period: {h_period}")
    # print(f"Maximum not periodical sequence length: {h_unique_len}")

    bins, hist = SequenceAnalyzer.hist(history)
    print(f"Expectation: {h_mean:.5f} (effective mean)")
    print(f"Standard deviation: {h_std:.5f} (corrected)")
    print(f"Variance: {h_var:.5f} (corrected)")
    plt.figure().add_subplot()
    plt.gca().hist(bins[:-1], bins, weights=hist)
    plt.show()


if __name__ == '__main__':
    while True:
        lab()
        print(f"Quit? [y]/n")
        if input() != "n":
            break
