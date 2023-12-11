from math import exp, ceil


cache: dict[float: float] = {}
GR_first_point, GR_second_point = (3 - 5**.5) / 2, (5**.5 - 1) / 2

GL_a, GL_b = 0, 1
epsilon = .0001


def function(x: float) -> float:
    return (1 + x**2)**.5 + exp(-2*x)


def function_first_der(x: float) -> float:
    return x / (1 + x**2)**.5 - 2 * exp(-2*x)


def function_second_der(x: float) -> float:
    return 1 / (1 + x**2)**1.5 + 4 * exp(-2*x)


def cache_function(func: callable(float)) -> callable(float):
    def res(x: float) -> float:
        global cache

        if x not in cache[func]:
            cache[func][x] = func(x)
        return cache[func][x]

    global cache

    cache[func] = {}
    return res


def passive_search(func: callable(float), segment: tuple[float, float], accuracy: float) -> float:
    def float_range(begin: float, end: float, step: float) -> list[float, ...]:
        n_steps = ceil((end - begin) / step)
        return [begin + j * step for j in range(n_steps + 1)]

    return min(float_range(*segment, accuracy*2), key=func)


def dichotomy_method(func: callable(float), segment: tuple[float, float], accuracy: float) -> float:
    a, b = segment
    delta = accuracy / 100

    accuracy *= 2
    while b - a > accuracy:
        d = (c := (a + b - delta) / 2) + delta
        if func(c) > func(d):
            a, b = c, b
        elif func(c) <= func(d):
            a, b = a, d

    return (a + b) / 2


def golden_ratio_method(func: callable(float), segment: tuple[float, float], accuracy: float) -> float:
    a, b = segment

    accuracy *= 2
    f_c, f_d = func(c := a + (b - a) * GR_first_point), func(d := a + (b - a) * GR_second_point)
    while b - a > accuracy:
        if f_c > f_d:
            a, b = c, b
            c, f_c, f_d = d, f_d, func(d := a + (b - a) * GR_second_point)
        elif f_c <= f_d:
            a, b = a, d
            d, f_d, f_c = c, f_c, func(c := a + (b - a) * GR_first_point)

    return (a + b) / 2


def fibonacci_sequence(fib_seq: list[int] = None, limit: float = None) -> list[int]:
    if fib_seq:
        return fibonacci_sequence([*fib_seq, fib_seq[-1] + fib_seq[-2]], limit) if fib_seq[-1] < limit else fib_seq
    else:
        return fibonacci_sequence([1, 1], .55 * (GL_b - GL_a) / epsilon)


def fibonacci_method(func: callable(float), segment: tuple[float, float], fib_seq: list[int], last_fib: int,
                     inner_point: tuple[int, float, float] = None) -> float:
    a, b = segment
    length = b - a

    if fib_seq == [1, 1, 2]:
        c = inner_point[1]
        d = c + .1 * length / last_fib
        if inner_point[2] > func(d):
            return (inner_point[1] + b) / 2
        else:
            return (a + d) / 2

    if inner_point is None:
        f_c = func(c := a + length * (fib_seq[-3] / fib_seq[-1]))
        f_d = func(d := a + length * (fib_seq[-2] / fib_seq[-1]))
    else:
        if inner_point[0] == 2:
            d, f_d = inner_point[1:]
            f_c = func(c := a + length * (fib_seq[-3] / fib_seq[-1]))
        else:
            c, f_c = inner_point[1:]
            f_d = func(d := a + length * (fib_seq[-2] / fib_seq[-1]))

    if f_c > f_d:
        return fibonacci_method(func, (c, b), fib_seq[:-1], last_fib, (1, d, f_d))
    else:
        return fibonacci_method(func, (a, d), fib_seq[:-1], last_fib, (2, c, f_c))


def tangent_method(func: callable(float), first_der: callable(float),
                   segment: tuple[float, float], accuracy: float) -> float:
    a, b = segment

    c = (func(b) - first_der(b) * b - func(a) + first_der(a) * a) / (first_der(a) - first_der(b))
    if abs(fc := first_der(c)) < accuracy:
        return c
    if fc < 0:
        a = c
    elif fc > 0:
        b = c

    return tangent_method(func, first_der, (a, b), accuracy)


def newton_raphson_method(first_der: callable(float), second_der: callable(float),
                          segment: tuple[float, float], accuracy: float) -> float:
    curr_x = segment[0]

    while abs(first_der(curr_x)) > accuracy:
        curr_x -= first_der(curr_x) / second_der(curr_x)

    return curr_x


def modified_newton_raphson_method(first_der: callable(float),
                                   segment: tuple[float, float], accuracy: float) -> float:
    curr_x, prev_x = segment

    while abs(first_der(curr_x)) > accuracy:
        prev_x, curr_x = (curr_x,
                          curr_x - first_der(curr_x) * (prev_x - curr_x) / (first_der(prev_x) - first_der(curr_x)))

    return curr_x


if __name__ == "__main__":
    print(passive_search(cache_function(function), (GL_a, GL_b), epsilon), len(cache[function]))
    print(dichotomy_method(cache_function(function), (GL_a, GL_b), epsilon), len(cache[function]))
    print(golden_ratio_method(cache_function(function), (GL_a, GL_b), epsilon), len(cache[function]))
    print(fibonacci_method(cache_function(function), (GL_a, GL_b), f := fibonacci_sequence(), f[-1]),
          len(cache[function]))
    print(tangent_method(cache_function(function), cache_function(function_first_der),
                         (GL_a, GL_b), epsilon), len(cache[function]), len(cache[function_first_der]))
    print(newton_raphson_method(
        cache_function(function_first_der),
        cache_function(function_second_der),
        (GL_a, GL_b), epsilon),
        0, len(cache[function_first_der]), len(cache[function_second_der]))
    print(modified_newton_raphson_method(
        cache_function(function_first_der),
        (GL_a, GL_b), epsilon),
        0, len(cache[function_first_der]))
