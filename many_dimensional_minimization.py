from vector import Vector
import math as m
from one_dimensional_minimization import cache_function, cache, golden_ratio_method
import numpy


DIM = 2


def function(x: Vector) -> float:
    return x[0] ** 2 + x[1] ** 2 - m.cos((x[0] - x[1]) / 2)


def grad(x: Vector) -> Vector:
    return Vector((2 * x[0] - .5 * m.sin((x[0] - x[1]) / 2), 2 * x[1] + .5 * m.sin((x[0] - x[1]) / 2)))


def hess(x: Vector) -> numpy.array:
    _x = (x[0] - x[1]) / 2
    return numpy.array([[2 - 1/4 * m.cos(_x), 1/4 * m.cos(_x)], [1/4 * m.cos(_x), 2 - 1/4 * m.cos(_x)]])


def coordinate_descent_method(func: callable(Vector), dim: int, step: float, step_reducer: float, accuracy: float):
    curr_x = Vector((1, 3))
    curr_step = step

    accuracy *= step_reducer / dim ** .5
    basis = [Vector.basis_vector(dim, j) for j in range(dim)]
    while curr_step >= accuracy:
        new_x = curr_x
        for e_j in basis:
            temp_x = new_x + curr_step * e_j
            if func(temp_x) < func(new_x):
                new_x = temp_x
                continue
            temp_x = new_x - curr_step * e_j
            if func(temp_x) < func(new_x):
                new_x = temp_x

        if curr_x == new_x:
            curr_step *= step_reducer
        else:
            curr_step = step
            curr_x = new_x

    return curr_x


def gradient_method_with_step_reducing(func: callable(Vector), gradient: callable(Vector), dim: int,
                                       step: float, step_reducer: float, epsilon: float, accuracy: float):
    curr_x = Vector((1, 3))
    curr_step = step

    _accuracy = accuracy * step_reducer / dim ** .5
    while curr_step > _accuracy:
        new_x = curr_x - curr_step * gradient(curr_x)
        if func(new_x) - func(curr_x) <= -curr_step * epsilon * abs(gradient(curr_x)) ** 2:
            curr_x = new_x
            curr_step = step
            if abs(gradient(curr_x)) <= accuracy:
                break
        else:
            curr_step *= step_reducer

    return curr_x


def descent_in_direction(func: callable(Vector), direction: Vector,
                         starting_point: Vector, accuracy: float) -> Vector:
    return starting_point - golden_ratio_method(cache_function(lambda a: func(starting_point - a * direction)),
                                                (-10, 10), accuracy) * direction


def steepest_gradient_descent_method(func: callable(Vector), gradient: callable(Vector), accuracy: float):
    curr_x = Vector((1, 3))

    while abs(gradient(curr_x)) > accuracy:
        curr_x = descent_in_direction(func, gradient(curr_x), curr_x, accuracy)

    return curr_x


def accelerated_gradient_descent_p_order(func: callable(Vector), gradient: callable(Vector), dim: int, accuracy: float):
    curr_x = Vector((1, 3))

    while abs(gradient(curr_x)) > accuracy:
        temp_x = curr_x
        for _ in range(dim):
            temp_x = descent_in_direction(func, gradient(temp_x), temp_x, accuracy)

        curr_x = descent_in_direction(func, temp_x - curr_x, curr_x, accuracy)

    return curr_x


def gully_method(func: callable(Vector), gradient: callable(Vector), accuracy: float):
    delta = Vector((accuracy ** .5, accuracy ** .5))
    curr_x = Vector((1, 3))

    while abs(gradient(curr_x)) > accuracy:
        temp_x = curr_x + delta
        curr_x = descent_in_direction(func, gradient(curr_x), curr_x, accuracy)
        temp_x = descent_in_direction(func, gradient(temp_x), temp_x, accuracy)
        curr_x = descent_in_direction(func, curr_x - temp_x, curr_x, accuracy)

    return curr_x


def newton_method(func: callable(Vector), gradient: callable(Vector), hessian: callable(Vector), accuracy: float):
    curr_x = Vector((1, 3))

    while abs(gradient(curr_x)) > accuracy:
        direction = -numpy.matmul(numpy.linalg.inv(hessian(curr_x)),
                                  curr_x.to_array())
        direction = Vector(tuple(x[0] for x in direction.tolist()))
        curr_x = descent_in_direction(func, direction, curr_x, accuracy)

    return curr_x


def quasi_newton_method(func: callable(Vector), gradient: callable(Vector), dim: int, accuracy: float):
    prev_x = Vector((1, 3))
    curr_x = descent_in_direction(func, gradient(prev_x), prev_x, accuracy)

    while abs(gradient(curr_x)) > accuracy:
        hessian = numpy.array([[1, 0], [0, 1]])

        for _ in range(dim):
            delta = (curr_x - prev_x).to_array()
            gamma = (gradient(curr_x) - gradient(prev_x)).to_array()
            v = delta - numpy.matmul(hessian, gamma)

            hessian = hessian + numpy.matmul(v, numpy.transpose(v)) / sum(x[0] for x in (v * gamma).tolist())

            direction = numpy.matmul(hessian, gradient(curr_x).to_array())
            direction = Vector(tuple(x[0] for x in direction.tolist()))
            curr_x, prev_x = descent_in_direction(func, direction, curr_x, accuracy), curr_x

        return curr_x


def conjugate_directions_method(func: callable(Vector), gradient: callable(Vector), dim: int, accuracy: float):
    curr_x = Vector((1, 3))

    while abs(gradient(curr_x)) > accuracy:
        d = -gradient(curr_x)
        for _ in range(dim):
            curr_x, prev_x = descent_in_direction(func, d, curr_x, accuracy), curr_x
            beta = abs(gradient(curr_x))**2 / abs(gradient(prev_x))**2
            d = -gradient(curr_x) + beta * d

    return curr_x


if __name__ == "__main__":
    print(coordinate_descent_method(cache_function(function), DIM, 1, .5, 1e-4),
          len(cache[function]))

    print(gradient_method_with_step_reducing(cache_function(function), cache_function(grad), DIM,
                                             1, .5, .1, 1e-4),
          len(cache[function]), len(cache[grad]))

    print(steepest_gradient_descent_method(cache_function(function), cache_function(grad), 1e-4),
          len(cache[function]), len(cache[grad]))

    print(accelerated_gradient_descent_p_order(cache_function(function), cache_function(grad), DIM, 1e-4),
          len(cache[function]), len(cache[grad]))

    print(gully_method(cache_function(function), cache_function(grad), 1e-4),
          len(cache[function]), len(cache[grad]))

    print(newton_method(cache_function(function), cache_function(grad), hess, 1e-4),
          len(cache[function]), len(cache[grad]))

    print(quasi_newton_method(cache_function(function), cache_function(grad), DIM, 1e-4),
          len(cache[function]), len(cache[grad]))

    print(conjugate_directions_method(cache_function(function), cache_function(grad), DIM, 1e-4),
          len(cache[function]), len(cache[grad]))
