from problem import ProblemBase, Sphere, Schwefel, Ackley, Rastrigin, Griewank
import numpy as np
from clustering import divide, to_group


class Swarm:
    def __init__(self, problem: ProblemBase, size: int = 50, n: int = 10, max_iter: int = 3000, avg_function='g',
                 chip_function='r',
                 density_function='g'):
        self.max_iterations = max_iter
        self.avg_function = avg_function
        self.chip_function = chip_function
        self.density_function = density_function
        self.c1 = 2.0
        self.c2 = 2.0
        self.n = n
        self.size = size
        self.problem = problem
        self.particles = np.array([Particle(self) for _ in range(size)])
        self.iteration = 0
        self.sub_swarms = None
        self.gBest = None
        self.gBestF = float('inf')
        self.update_particle_best()

    def omega(self):
        return 0.9 - (0.9 - 0.4) * self.iteration / self.max_iterations

    def iterate(self):
        self.sub_swarms = self.divide_sub_swarms()
        for sub_swarm in self.sub_swarms:
            sub_swarm.update()
        self.update_particle_best()
        self.iteration += 1

    def divide_sub_swarms(self):
        label, _ = divide(np.array([particle.loc for particle in self.particles]), self.n,
                          density_function=self.density_function)
        sub_swarms = [SubSwarm(self, group) for group in to_group(label, self.n)]
        return sub_swarms

    def update_particle_best(self):
        for index in range(self.size):
            if self.particles[index].pBestF < self.gBestF:
                self.gBest = self.particles[index].pBest
                self.gBestF = self.particles[index].pBestF

    def find_best_particles(self, indexes):
        best_value = float('inf')
        best_index = -1
        for index in indexes:
            if self.particles[index].fitness < best_value:
                best_index = index
        return best_index

    def calculate_best(self) -> np.ndarray:
        if self.avg_function == 'c':
            return np.average([sub_swarm.cgBest for sub_swarm in self.sub_swarms])
        elif self.avg_function == 'g':
            return self.gBest
        else:
            raise Exception('avg_function must be \'c\' or \'g\'')

    def evolute(self, debug: bool = True, draw: bool = True):
        history = np.zeros(self.max_iterations)
        while self.iteration < self.max_iterations:
            self.iterate()
            history[self.iteration - 1] = self.gBestF
            if debug:
                print(f"[{self.iteration}/{self.max_iterations}] {self.gBestF}")
        print(self.gBestF)
        if draw:
            from matplotlib import pyplot as plt
            plt.figure(dpi=300)
            plt.xlabel('iterations')
            plt.ylabel('lg(Error)')
            plt.plot(range(1, len(history) + 1), [-323 if err == 0 else np.log10(err) for err in history])
            plt.show()
        return history


class SubSwarm:
    def __init__(self, swarm: Swarm, sub_particles_index: list):
        self.sub_particles_index = sub_particles_index
        self.swarm = swarm
        self.cgBestIndex = self.swarm.find_best_particles(self.sub_particles_index)
        self.cgBest = self.swarm.particles[self.cgBestIndex].loc.copy()

    def update(self):
        self.swarm.particles[self.cgBestIndex].update_local_best()
        self.swarm.particles[self.cgBestIndex].update_status()
        for index in self.sub_particles_index:
            particle = self.swarm.particles[index]
            if index != self.cgBestIndex:
                particle.update_ordinary(self)
                particle.update_status()


class Particle:
    def __init__(self, swarm: Swarm):
        self.swarm = swarm
        self.loc: np.ndarray = swarm.problem.random()
        self.fitness = swarm.problem.invoke(self.loc)
        self.pBest = self.loc.copy()
        self.pBestF = self.fitness

    def update_local_best(self):
        self.loc = \
            self.swarm.omega() * self.loc \
            + self.swarm.c1 * np.random.uniform(0, 1) * (self.pBest - self.loc) \
            + self.swarm.c2 * np.random.uniform(0, 1) * (
                    self.swarm.calculate_best() - self.loc)

    def update_ordinary(self, sub_swarm: SubSwarm):
        self.loc = self.swarm.omega() * self.loc \
                   + self.swarm.c1 * np.random.uniform(0, 1) * (self.pBest - self.loc) \
                   + self.swarm.c2 * np.random.uniform(0, 1) * (sub_swarm.cgBest - self.loc)

    def update_status(self):
        self.loc = self.swarm.problem.clip(self.loc, clip_function=self.swarm.chip_function)
        self.fitness = self.swarm.problem.invoke(self.loc)
        if self.fitness < self.pBestF:
            self.pBest = self.loc.copy()
            self.pBestF = self.fitness


if __name__ == '__main__':
    swarm_ = Swarm(Schwefel(), size=20, n=4, avg_function='c', chip_function='c', density_function='g',
                   max_iter=10000)
    swarm_.evolute(debug=True, draw=True)

    print(swarm_.gBest)
