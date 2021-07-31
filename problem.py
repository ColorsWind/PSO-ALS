import numpy as np


class ProblemBase(object):
    def __init__(self, dimension: int, lower: float, upper: float):
        self.dimension = dimension
        self.lower = lower
        self.upper = upper

    def invoke(self, array) -> np.float64:
        pass

    def random(self) -> np.ndarray:
        return np.random.uniform(self.lower, self.upper, self.dimension)

    def clip(self, array, clip_function='c'):
        if clip_function == 'c':
            return np.clip(array, self.lower, self.upper)
        elif clip_function == 'r':
            for value in array:
                if value < self.lower or value > self.upper:
                    return self.random()
            return array
        else:
            raise Exception('clip_function must be \'c\' or \'r\'')

    def draw(self):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        X = np.linspace(self.lower, self.upper, 100)
        Y = np.linspace(self.lower, self.upper, 100)
        dimension = self.dimension
        self.dimension = 2
        Z = np.array([[self.invoke(np.array([x, y])) for y in Y] for x in X])
        self.dimension = dimension
        fig = plt.figure(dpi=300)
        ax = Axes3D(fig)
        ax.set_zlabel('z')
        plt.title(self.__class__.__name__)
        plt.xlabel('x')
        plt.ylabel('y')
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.show()


class Sphere(ProblemBase):
    def __init__(self):
        ProblemBase.__init__(self, 30, -100, 100)

    def invoke(self, array) -> np.float64:
        return np.sum(array * array)


class Schwefel(ProblemBase):
    def __init__(self):
        ProblemBase.__init__(self, 10000, -500, 500)

    def invoke(self, array) -> np.float64:
        return np.sum(-array * np.sin(np.sqrt(np.abs(array)))) + self.dimension * 418.9829


class Ackley(ProblemBase):
    def __init__(self):
        ProblemBase.__init__(self, 30, -32, 32)

    def invoke(self, array) -> np.float64:
        return -20 * np.exp(-0.2 * np.sqrt(1 / self.dimension * np.sum(array ** 2))) \
               - np.exp(1 / self.dimension * np.sum(np.cos(2 * np.pi * array))) + 20 + np.e


class Rastrigin(ProblemBase):
    def __init__(self):
        ProblemBase.__init__(self, 30, -5.12, 5.12)

    def invoke(self, array) -> np.float64:
        return np.sum(array ** 2 - 10 * np.cos(2 * np.pi * array)) + self.dimension * 10


class Griewank(ProblemBase):
    def __init__(self):
        ProblemBase.__init__(self, 30, -600, 600)

    def invoke(self, array) -> np.float64:
        value = 1
        for i in range(self.dimension):
            value *= np.cos(array[i] / (i + 1))
        return np.sum(array * array) / 4000 + value + 1


if __name__ == '__main__':
    Schwefel().draw()
