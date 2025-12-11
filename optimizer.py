import numpy as np
import math
import random
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes
from crypto_engine import CryptoBenchmarkV4

# Deterministik Test
random.seed(42)
np.random.seed(42)

class AdvancedOptimizer:
    def __init__(self, pop_size=10, max_iter=20, data_size_mb=2):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.engine = CryptoBenchmarkV4()
        
        # 2MB Veri (Vize için güvenli limit, throttling riskini azaltır)
        print(f">>> {data_size_mb} MB test verisi hazırlanıyor...")
        self.test_data = get_random_bytes(data_size_mb * 1024 * 1024)
        
        # Parametreler: [Algo, Mod, Key, Buffer] (Thread YOK!)
        # Algo(0-4), Mod(0-2), Key(0-2), Buffer(1KB-64KB)
        self.lb = [0, 0, 0, 1024]
        self.ub = [4.49, 2.49, 2.49, 65536]
        self.dim = 4 # Boyut düşürüldü
        
        self.convergence_curve = []
        
        # Min-Max Normalizasyon için Baseline
        print(">>> Baseline (Min-Max) analizi yapılıyor...")
        self.limits = self._collect_limits(samples=15)
        print(f">>> Limitler: {self.limits}")

    def _collect_limits(self, samples):
        """Ortamın Min ve Max değerlerini öğrenir."""
        # Başlangıç değerleri ters verilir
        lims = {
            "Min_Time": 1e9, "Max_Time": 0,
            "Min_CPU": 1e9, "Max_CPU": 0,
            "Min_Mem": 1e9, "Max_Mem": 0
        }
        
        for _ in range(samples):
            rand_x = [random.uniform(self.lb[j], self.ub[j]) for j in range(self.dim)]
            res = self.engine._run_single_test(rand_x, self.test_data)
            
            if res:
                # Max güncelle
                if res["Time"] > lims["Max_Time"]: lims["Max_Time"] = res["Time"]
                if res["CPU_Time"] > lims["Max_CPU"]: lims["Max_CPU"] = res["CPU_Time"]
                if res["Memory"] > lims["Max_Mem"]: lims["Max_Mem"] = res["Memory"]
                # Min güncelle
                if res["Time"] < lims["Min_Time"]: lims["Min_Time"] = res["Time"]
                if res["CPU_Time"] < lims["Min_CPU"]: lims["Min_CPU"] = res["CPU_Time"]
                if res["Memory"] < lims["Min_Mem"]: lims["Min_Mem"] = res["Memory"]
        
        # Safety Margin
        for k in lims:
            if "Max" in k: lims[k] *= 1.1
            if "Min" in k: lims[k] *= 0.9
        
        return lims

    def normalize(self, val, min_v, max_v):
        if max_v == min_v: return 1.0
        norm = (val - min_v) / (max_v - min_v)
        return min(max(norm, 0.0), 1.0) # Clamp

    def calculate_fitness(self, x):
        res = self.engine.benchmark_with_stats(x, self.test_data, repeats=5)
        if res is None: return self.engine.PENALTY
        
        # Min-Max Normalizasyon (Bilimsel Yöntem)
        n_Time = self.normalize(res["Time"], self.limits["Min_Time"], self.limits["Max_Time"])
        n_CPU = self.normalize(res["CPU_Time"], self.limits["Min_CPU"], self.limits["Max_CPU"])
        n_Mem = self.normalize(res["Memory"], self.limits["Min_Mem"], self.limits["Max_Mem"])
        n_Sec = res["Security"]
        
        cost = (self.engine.wT * n_Time) + \
               (self.engine.wCPU * n_CPU) + \
               (self.engine.wM * n_Mem) - \
               (self.engine.wS * n_Sec)
        return cost

    def optimize(self):
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.lb[j], self.ub[j])

        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")

        print("-" * 60)
        print(f"{'Iter':<5} | {'Cost':<10} | {'Algoritma'} | {'Mod'}")
        print("-" * 60)

        for t in range(self.max_iter):
            # 1. Fitness Update
            for i in range(self.pop_size):
                population[i, :] = np.clip(population[i, :], self.lb, self.ub)
                fitness = self.calculate_fitness(population[i, :])
                
                if fitness < rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = population[i, :].copy()
            
            self.convergence_curve.append(rabbit_score)
            
            # 2. HHO-WOA Hibrit Mantığı
            # Enerji faktörü
            E1 = 2 * (1 - (t / self.max_iter))
            
            # Adaptif Ağırlık (HHO -> WOA geçişi)
            # Alpha zamanla azalır (0.9 -> 0.1), böylece WOA şansı artar
            alpha = 0.9 * (1 - (t / self.max_iter))

            for i in range(self.pop_size):
                E0 = 2 * random.random() - 1
                E = 2 * E0 * E1
                
                # --- HHO FAZI (Exploration & Besiege) ---
                if random.random() < alpha + 0.1: 
                    if abs(E) >= 1: # Exploration
                        q = random.random()
                        rand_idx = random.randint(0, self.pop_size-1)
                        if q < 0.5:
                            population[i, :] = population[rand_idx, :] - random.random() * abs(population[rand_idx, :] - 2 * random.random() * population[i, :])
                        else:
                            population[i, :] = (rabbit_pos - population.mean(0)) - random.random() * ((np.array(self.ub) - np.array(self.lb)) * random.random() + np.array(self.lb))
                    else: 
                        # Exploitation (Soft/Hard Besiege Ayrımı - Basitleştirilmiş)
                        r = random.random()
                        if r >= 0.5 and abs(E) >= 0.5: # Soft Besiege
                            J = 2 * (1 - random.random())
                            population[i, :] = (rabbit_pos - population[i, :]) - E * abs(J * rabbit_pos - population[i, :])
                        elif r >= 0.5 and abs(E) < 0.5: # Hard Besiege
                             population[i, :] = (rabbit_pos - population[i, :]) - E * abs(rabbit_pos - population[i, :])
                        else: # Rapid Dives (WOA benzeri saldırı)
                             population[i, :] = (rabbit_pos - population[i, :]) - E * abs(rabbit_pos - population[i, :]) + random.random()
                
                # --- WOA FAZI (Spiral Attack) ---
                else:
                    distance = abs(rabbit_pos - population[i, :])
                    b = 1
                    l = (random.random() * 2) - 1
                    population[i, :] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + rabbit_pos

            # Loglama
            algo_map = ["AES", "ChaCha", "3DES", "Blow", "CAST"]
            mod_map = ["CBC", "GCM/AEAD", "CTR"]
            
            a_idx = int(round(rabbit_pos[0]))
            m_idx = int(round(rabbit_pos[1]))
            a_name = algo_map[min(a_idx, 4)]
            m_name = mod_map[min(m_idx, 2)]
            
            print(f"{t+1:<5} | {rabbit_score:.5f}    | {a_name:<9} | {m_name}")

        return rabbit_pos, rabbit_score

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'g-o', linewidth=2, label="V4 Hybrid HHO-WOA")
        plt.title('Bilimsel Optimizasyon (V4)', fontsize=14)
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('Normalize Maliyet', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig('v4_result.png')
        print(">>> Grafik kaydedildi: v4_result.png")

if __name__ == "__main__":
    # Vize için 2MB veri yeterli.
    optimizer = AdvancedOptimizer(pop_size=8, max_iter=15, data_size_mb=2)
    best_x, best_score = optimizer.optimize()
    optimizer.plot_results()
    
    print("\n>>> GLOBAL OPTIMUM (V4) <<<")
    print(f"Skor: {best_score}")
    print(f"Parametreler: {best_x}")