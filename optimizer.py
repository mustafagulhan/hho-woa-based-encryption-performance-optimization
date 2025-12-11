import numpy as np
import math
import random
import matplotlib.pyplot as plt
from crypto_engine import CryptoBenchmark

class HybridOptimizer:
    def __init__(self, pop_size=10, max_iter=20):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.engine = CryptoBenchmark()
        
        # PARAMETRE SINIRLARI
        # x1: Algo (0-4), x2: Mod (0-2), x3: Key (0-2), x4: Block (1024-65536), x5: Thread (1-8)
        self.lb = [0, 0, 0, 1024, 1]
        self.ub = [4.99, 2.99, 2.99, 65536, 8]
        self.dim = 5 
        self.convergence_curve = [] 

    def optimize(self):
        population = np.zeros((self.pop_size, self.dim))
        
        # Rastgele Başlangıç
        for i in range(self.pop_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.lb[j], self.ub[j])

        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")

        print(f"{'İterasyon':<10} | {'En İyi Skor':<15} | {'En İyi Parametreler'}")
        print("-" * 75)

        for t in range(self.max_iter):
            # --- Fitness Hesaplama ---
            for i in range(self.pop_size):
                for j in range(self.dim):
                    population[i, j] = np.clip(population[i, j], self.lb[j], self.ub[j])
                
                fitness = self.engine.fitness(population[i, :])
                
                if fitness < rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = population[i, :].copy()

            self.convergence_curve.append(rabbit_score)

            # --- HHO-WOA Güncelleme ---
            E1 = 2 * (1 - (t / self.max_iter))
            for i in range(self.pop_size):
                E0 = 2 * random.random() - 1
                E = 2 * E0 * (1 - (t / self.max_iter))
                p = random.random()
                
                if p < 0.5: # HHO
                    if abs(E) >= 1:
                        q = random.random()
                        rand_hawk_idx = random.randint(0, self.pop_size - 1)
                        rand_hawk = population[rand_hawk_idx, :]
                        if q < 0.5:
                            population[i, :] = rand_hawk - random.random() * abs(rand_hawk - 2 * random.random() * population[i, :])
                        else:
                            population[i, :] = (rabbit_pos - population[i, :].mean(0)) - random.random() * ((self.ub[0] - self.lb[0]) * random.random() + self.lb[0])
                    else:
                        J = 2 * (1 - random.random())
                        population[i, :] = (rabbit_pos - population[i, :]) - E * abs(J * rabbit_pos - population[i, :])
                else: # WOA
                    distance = abs(rabbit_pos - population[i, :])
                    b = 1
                    l = (random.random() * 2) - 1
                    population[i, :] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + rabbit_pos

            display_params = [int(round(x)) for x in rabbit_pos]
            print(f"{t+1:<10} | {rabbit_score:.5f}         | {display_params}")

        return rabbit_pos, rabbit_score

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'r-o', linewidth=2, label="HHO-WOA")
        plt.title('Optimizasyon Süreci (Yakınsama)', fontsize=14)
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('Maliyet (Daha düşük daha iyi)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig('grafik_yakinsama.png')
        print("-> Yakınsama grafiği kaydedildi: grafik_yakinsama.png")
        # plt.show() # Pencereleri üst üste açmamak için kapalı tutuyoruz

    def compare_all_algorithms(self, best_params):
        """
        Bulunan en iyi ayarları (Blok boyutu, Thread vb.) sabit tutarak
        tüm algoritmaları birbiriyle kıyaslar.
        """
        algos = ["AES", "ChaCha20", "3DES", "Blowfish", "CAST5"]
        scores = []
        
        print("\n" + "="*50)
        print("FİNAL KIYASLAMA TESTİ (Aynı Şartlar Altında)")
        print("="*50)
        print(f"{'Algoritma':<15} | {'Maliyet (Cost)':<15} | {'Durum'}")
        print("-" * 50)

        # HHO-WOA'nın bulduğu parametreleri al
        best_mode = best_params[1]
        best_key = best_params[2]
        best_block = best_params[3]
        best_thread = best_params[4]

        # Her algoritma için testi çalıştır
        for i in range(5):
            # Parametre setini oluştur: [AlgoID, Mode, Key, Block, Thread]
            test_params = [i, best_mode, best_key, best_block, best_thread]
            
            # Motoru çalıştır
            score = self.engine.fitness(test_params)
            scores.append(score)
            
            # Kazananı işaretle
            status = "KAZANAN (OPTIMUM)" if score == min(scores) else ""
            if i > 0 and score == min(scores): status = "KAZANAN" # İlk değilse

            print(f"{algos[i]:<15} | {score:.5f}         | {status}")

        # --- KIYASLAMA GRAFİĞİ (BAR CHART) ---
        plt.figure(figsize=(10, 6))
        colors = ['gray'] * 5
        # En düşük skora sahip olanı (Kazananı) Yeşil yap
        best_idx = scores.index(min(scores))
        colors[best_idx] = 'green'
        
        plt.bar(algos, scores, color=colors)
        plt.title('Algoritmaların Performans Kıyaslaması\n(Sabit Blok Boyutu ve Thread Ayarlarında)', fontsize=14)
        plt.ylabel('Maliyet Skoru (Düşük olan iyi)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig('grafik_kiyaslama.png')
        print("-> Kıyaslama grafiği kaydedildi: grafik_kiyaslama.png")
        plt.show()

# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    print("\n>>> HHO-WOA Optimizasyonu Başlıyor...\n")
    
    optimizer = HybridOptimizer(pop_size=15, max_iter=20)
    best_params, best_score = optimizer.optimize()
    
    # 1. Yakınsama Grafiğini Çiz
    optimizer.plot_convergence()

    # 2. Sonuç Raporu
    print("\n" + "="*50)
    print("OPTIMIZASYON SONUCU")
    print("="*50)
    algos = ["AES", "ChaCha20", "3DES", "Blowfish", "CAST5"]
    algo_name = algos[int(round(best_params[0]))]
    print(f"Seçilen Algoritma : {algo_name}")
    print(f"Blok Boyutu       : {int(round(best_params[3]))} bytes")
    print(f"Thread Sayısı     : {int(round(best_params[4]))}")

    # 3. Kıyaslama (Comparison) Yap
    # İşte senin istediğin "Hepsini deneyip sonuç gösterme" kısmı:
    optimizer.compare_all_algorithms(best_params)