import numpy as np
import math
import random
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes
from crypto_engine import CryptoBenchmarkV2

class AcademicOptimizer:
    def __init__(self, pop_size=10, max_iter=20, data_size_mb=10):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.engine = CryptoBenchmarkV2()
        self.data_size_bytes = data_size_mb * 1024 * 1024
        
        # Test Verisi (Gerçek veri, simülasyon yok)
        print(f">>> {data_size_mb} MB test verisi oluşturuluyor...")
        self.test_data = get_random_bytes(self.data_size_bytes)
        print(">>> Veri hazır.")

        # Parametre Sınırları
        # x0: Algo(0-4), x1: Mod(0-2), x2: Key(0-2), x3: Buffer(1KB-64KB), x4: Dummy(Thread)
        self.lb = [0, 0, 0, 1024, 1]
        self.ub = [4.49, 2.49, 2.49, 65536, 8]
        self.dim = 5
        
        self.convergence_curve = []
        
        # Dinamik Normalizasyon için Geçmiş Kaydı
        self.history_max = {"Time": 1.0, "CPU": 1.0, "Mem": 1.0}

    def calculate_fitness(self, x):
        res = self.engine.run_benchmark(x, self.test_data)
        if res is None:
            return self.engine.PENALTY
        
        # --- DİNAMİK NORMALİZASYON (Kritik Düzeltme) ---
        # Gördüğümüz en yüksek değerleri güncelliyoruz
        if res["Time"] > self.history_max["Time"]: self.history_max["Time"] = res["Time"]
        if res["CPU_Time"] > self.history_max["CPU"]: self.history_max["CPU"] = res["CPU_Time"]
        if res["Memory"] > self.history_max["Mem"]: self.history_max["Mem"] = res["Memory"]
        
        # 0-1 Arasına çekme
        n_Time = res["Time"] / self.history_max["Time"]
        n_CPU = res["CPU_Time"] / self.history_max["CPU"]
        n_Mem = res["Memory"] / self.history_max["Mem"]
        n_Sec = res["Security"] # Zaten 0-1 arası
        
        # Amaç Fonksiyonu
        cost = (self.engine.wT * n_Time) + \
               (self.engine.wCPU * n_CPU) + \
               (self.engine.wM * n_Mem) - \
               (self.engine.wS * n_Sec)
               
        return cost

    def optimize(self):
        population = np.zeros((self.pop_size, self.dim))
        
        # Başlangıç Popülasyonu
        for i in range(self.pop_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.lb[j], self.ub[j])

        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")

        print(f"{'İter':<5} | {'Cost':<10} | {'Parametreler (Algo-Mod-Key-Buf)'}")
        print("-" * 60)

        for t in range(self.max_iter):
            
            # 1. Fitness Hesapla & Rabbit Güncelle
            for i in range(self.pop_size):
                # Sınır Kontrolü
                population[i, :] = np.clip(population[i, :], self.lb, self.ub)
                
                fitness = self.calculate_fitness(population[i, :])
                
                if fitness < rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = population[i, :].copy()
            
            self.convergence_curve.append(rabbit_score)
            
            # 2. HHO-WOA Hibrit Döngüsü (Akademik Versiyon)
            # E1: Enerjinin azalması (Lineer değil, Non-lineer olabilir ama HHO lineer kullanır)
            E1 = 2 * (1 - (t / self.max_iter))
            
            for i in range(self.pop_size):
                E0 = 2 * random.random() - 1
                E = 2 * E0 * E1  # Kaçış Enerjisi
                
                # --- FAZ 1: KEŞİF (EXPLORATION) - HHO KULLANILIR ---
                if abs(E) >= 1:
                    # HHO Exploration Equations
                    q = random.random()
                    rand_hawk_idx = random.randint(0, self.pop_size - 1)
                    rand_hawk = population[rand_hawk_idx, :]
                    
                    if q < 0.5:
                        # Random konuma göre güncelle
                        population[i, :] = rand_hawk - random.random() * abs(rand_hawk - 2 * random.random() * population[i, :])
                    else:
                        # Tavşan ve ortalama konuma göre güncelle
                        population[i, :] = (rabbit_pos - population.mean(0)) - random.random() * ((np.array(self.ub) - np.array(self.lb)) * random.random() + np.array(self.lb))
                
                # --- FAZ 2: SÖMÜRÜ (EXPLOITATION) - WOA ENTEGRASYONU ---
                else:
                    # Burada HHO'nun "Besiege" taktikleri yerine
                    # WOA'nın meşhur "Spiral Updating" mekanizmasını kullanıyoruz.
                    # Bu, literatürde "HHO with WOA Mutation" veya "Hybrid HHO-WOA" olarak geçer.
                    
                    # Spiral Denklem (WOA)
                    distance = abs(rabbit_pos - population[i, :])
                    b = 1 # Spiral sabit
                    l = (random.random() * 2) - 1 # -1 ile 1 arası
                    
                    # D' * e^bl * cos(2*pi*l) + Rabbit
                    population[i, :] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + rabbit_pos

            # İlerleme Raporu
            algo_map = ["AES", "ChaCha", "3DES", "Blow", "CAST"]
            algo_name = algo_map[int(round(rabbit_pos[0]))]
            print(f"{t+1:<5} | {rabbit_score:.5f}    | {algo_name} - {int(rabbit_pos[3])} byte Buf")

        return rabbit_pos, rabbit_score

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'r-o', linewidth=2, label="Hybrid HHO-WOA (Academic)")
        plt.title('Yakınsama Grafiği (Dinamik Normalizasyon)', fontsize=14)
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('Maliyet', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig('academic_result.png')
        print("Grafik kaydedildi.")

if __name__ == "__main__":
    # Gerçek veri işlediğimiz için 10MB veya 20MB makul bir testtir.
    # 50 MB biraz uzun sürebilir ama denenebilir.
    optimizer = AcademicOptimizer(pop_size=10, max_iter=15, data_size_mb=10)
    best_x, best_score = optimizer.optimize()
    optimizer.plot_results()
    
    print("\n>>> EN İYİ ÇÖZÜM <<<")
    print(f"Skor: {best_score}")
    print(f"Parametreler: {best_x}")