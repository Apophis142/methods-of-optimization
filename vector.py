import numpy


class Vector(object):
    def __init__(self, coordinates: tuple[float, ...]):
        self.coordinates = tuple(coordinates)
        self.dim = len(coordinates)

    def __add__(self, other):
        if isinstance(other, Vector) and self.dim == other.dim:
            return Vector(tuple(self.coordinates[j] + other.coordinates[j] for j in range(self.dim)))

        raise TypeError

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Vector(tuple(x * other for x in self.coordinates))

        raise TypeError

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return sum(x ** 2 for x in self.coordinates) ** .5

    def __getitem__(self, item):
        return self.coordinates[item]

    def __eq__(self, other):
        if isinstance(other, Vector) and self.dim == other.dim:
            return all(self[j] == other[j] for j in range(self.dim))
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return self.coordinates.__hash__()

    def __str__(self):
        return '(' + ' '.join(map(str, self.coordinates)) + ')'

    def __repr__(self):
        return self.__str__()

    def to_array(self) -> numpy.array:
        return numpy.array([[x] for x in self.coordinates])

    @classmethod
    def zero(cls, dim: int):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError
        return cls(tuple([0] * dim))

    @classmethod
    def basis_vector(cls, dim: int, j: int):
        if isinstance(dim, int) and isinstance(j, int) and 0 <= j < dim:
            return cls(tuple([0] * j + [1] + [0] * (dim - j - 1)))
        raise ValueError
