import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Tekrarlanabilirlik
random.seed(42)
np.random.seed(42)


def michalewicz(x: np.ndarray, m: int = 10) -> float:
    """
    Michalewicz fonksiyonu (minimizasyon).
    Aralık: 0 <= x_i <= pi
    """
    i = np.arange(1, x.size + 1)
    return -np.sum(np.sin(x) * (np.sin(i * (x ** 2) / math.pi) ** (2 * m)))


class MichalewiczOptimizer:
    """
    HHO-WOA, DE ve PSO'yu Michalewicz fonksiyonu için karşılaştırmalı olarak çalıştırır.
    """

    def __init__(self, dim: int = 50, pop_size: int = 60, max_iter: int = 1000, m_param: int = 10):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.m_param = m_param
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim) * math.pi
        self.curves: Dict[str, List[float]] = {}

    # Ortak yardımcılar
    def _init_population(self) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))

    def _fitness(self, x: np.ndarray) -> float:
        return michalewicz(x, m=self.m_param)

    # HHO-WOA
    def run_hho_woa(self) -> Tuple[np.ndarray, float]:
        population = self._init_population()
        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")
        curve: List[float] = []

        print("-" * 90)
        print(f"{'Iter':<6} | {'Cost':<12} | {'Note'}")
        print("-" * 90)

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                population[i] = np.clip(population[i], self.lb, self.ub)
                fitness = self._fitness(population[i])
                if fitness < rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = population[i].copy()

            curve.append(rabbit_score)

            alpha = 1 - (t / self.max_iter)
            E1 = 2 * alpha

            for i in range(self.pop_size):
                if random.random() < alpha:
                    E0 = 2 * random.random() - 1
                    E = 2 * E0 * E1
                    if abs(E) >= 1:
                        q = random.random()
                        rand_idx = random.randint(0, self.pop_size - 1)
                        if q < 0.5:
                            population[i] = population[rand_idx] - random.random() * abs(
                                population[rand_idx] - 2 * random.random() * population[i]
                            )
                        else:
                            population[i] = (rabbit_pos - population.mean(0)) - random.random() * (
                                (self.ub - self.lb) * random.random() + self.lb
                            )
                    else:
                        population[i] = (rabbit_pos - population[i]) - E * abs(rabbit_pos - population[i])
                else:
                    distance = abs(rabbit_pos - population[i])
                    b = 1
                    l = (random.random() * 2) - 1
                    population[i] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + rabbit_pos

            if (t + 1) % 50 == 0 or t == 0:
                print(f"{t+1:<6} | {rabbit_score:<12.6f} | best-so-far")

        self.curves["HHO-WOA"] = curve
        return rabbit_pos, rabbit_score

    # Differential Evolution
    def run_de(self, F: float = 0.5, CR: float = 0.9) -> Tuple[np.ndarray, float]:
        population = self._init_population()
        fitness = np.array([self._fitness(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_score = fitness[best_idx]
        curve: List[float] = []

        print("-" * 90)
        print(">>> DE başlıyor...")
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[random.sample(idxs, 3)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = self._fitness(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_score:
                        best_score = f_trial
                        best_pos = trial.copy()
            curve.append(best_score)
            if (t + 1) % 50 == 0 or t == 0:
                print(f"DE iter {t+1}/{self.max_iter} | best {best_score:.6f}")

        self.curves["DE"] = curve
        return best_pos, best_score

    # Particle Swarm Optimization
    def run_pso(self, w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Tuple[np.ndarray, float]:
        population = self._init_population()
        velocity = np.zeros_like(population)
        fitness = np.array([self._fitness(ind) for ind in population])

        pbest_pos = population.copy()
        pbest_score = fitness.copy()
        best_idx = np.argmin(fitness)
        gbest_pos = population[best_idx].copy()
        gbest_score = fitness[best_idx]
        curve: List[float] = []

        print("-" * 90)
        print(">>> PSO başlıyor...")
        for t in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocity = (w * velocity) + c1 * r1 * (pbest_pos - population) + c2 * r2 * (gbest_pos - population)
            population = population + velocity
            population = np.clip(population, self.lb, self.ub)

            for i in range(self.pop_size):
                f_val = self._fitness(population[i])
                if f_val < pbest_score[i]:
                    pbest_score[i] = f_val
                    pbest_pos[i] = population[i].copy()
                    if f_val < gbest_score:
                        gbest_score = f_val
                        gbest_pos = population[i].copy()
            curve.append(gbest_score)
            if (t + 1) % 50 == 0 or t == 0:
                print(f"PSO iter {t+1}/{self.max_iter} | best {gbest_score:.6f}")

        self.curves["PSO"] = curve
        return gbest_pos, gbest_score

    def plot_results(self, filename: str = "michalewicz_result.png"):
        plt.figure(figsize=(10, 6))
        for name, curve in self.curves.items():
            plt.plot(curve, linewidth=2, label=name)
        plt.title("Michalewicz (d=50, m=10) - HHO/DE/PSO", fontsize=14)
        plt.xlabel("Iterasyon", fontsize=12)
        plt.ylabel("Maliyet (düşük daha iyi)", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        print(f">>> Grafik kaydedildi: {filename}")

    def summarize(self, label: str, x: np.ndarray, score: float):
        print(f"[{label}] En İyi Skor: {score:.6f}")
        print("  x[0:5] (örnek) :", np.round(x[:5], 4))
        print("  x[5:10] (örnek):", np.round(x[5:10], 4))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(">>> Michalewicz (d=50, m=10) Optimizasyon Başlıyor <<<")
    print("=" * 70)

    optimizer = MichalewiczOptimizer(dim=50, pop_size=60, max_iter=1000, m_param=10)

    hho_x, hho_score = optimizer.run_hho_woa()
    de_x, de_score = optimizer.run_de()
    pso_x, pso_score = optimizer.run_pso()

    optimizer.plot_results()

    print("\n" + "=" * 70)
    print("🏆 Michalewicz Sonuç Özeti (d=50, m=10)")
    print("=" * 70)
    optimizer.summarize("HHO-WOA", hho_x, hho_score)
    print("-" * 60)
    optimizer.summarize("DE", de_x, de_score)
    print("-" * 60)
    optimizer.summarize("PSO", pso_x, pso_score)
    print("-" * 60)
    print(">>> Analiz Tamamlandı.")

