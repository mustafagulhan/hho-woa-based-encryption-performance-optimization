import time
import os
import psutil
import numpy as np
from Crypto.Cipher import AES, ChaCha20, DES3, Blowfish, CAST
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

class CryptoBenchmark:
    def __init__(self):
        # Amaç fonksiyonu ağırlıkları
        self.wT = 0.4    # Süre
        self.wCPU = 0.3  # CPU
        self.wM = 0.1    # RAM
        self.wS = 0.2    # Güvenlik (Çıkarılacak)
        
        # Ceza Puanı
        self.PENALTY = 9999.0

    def get_security_score(self, algo_name, key_size):
        score = 0.5 # Baz puan
        
        # Anahtar uzunluğu katkısı
        if key_size >= 32: score += 0.4  # 256-bit
        elif key_size >= 24: score += 0.2 # 192-bit
        elif key_size >= 16: score += 0.1 # 128-bit
        
        # Algoritma türü katkısı
        if algo_name == "ChaCha20": score += 0.1
        if algo_name == "3DES": score -= 0.2
        if algo_name == "Blowfish": score -= 0.1
        
        return min(score, 1.0)

    def run_encryption(self, params, data):
        """
        params: [Algoritma_ID, Mod_ID, Key_Size_Index, Block_Size, Threads]
        data: Şifrelenecek ham veri (bytes)
        """
        # Mapping (Sayıları ayarlara çevirme)
        algo_id = int(round(params[0]))
        mode_id = int(round(params[1]))
        key_idx = int(round(params[2]))
        
        key_sizes = [16, 24, 32] 
        key_idx = max(0, min(key_idx, 2))
        key_len = key_sizes[key_idx]

        cipher = None
        algo_name = ""

        try:
            # --- ALGORİTMA SEÇİMİ ---
            if algo_id == 0: # AES
                algo_name = "AES"
                selected_mode = AES.MODE_GCM if mode_id > 0.5 else AES.MODE_CBC
                key = get_random_bytes(key_len)
                cipher = AES.new(key, selected_mode)
            
            elif algo_id == 1: # ChaCha20
                algo_name = "ChaCha20"
                key = get_random_bytes(32)
                cipher = ChaCha20.new(key=key)
            
            elif algo_id == 2: # 3DES
                algo_name = "3DES"
                key_len_3des = 16 if key_idx == 0 else 24 
                key = get_random_bytes(key_len_3des)
                cipher = DES3.new(key, DES3.MODE_CBC)
            
            elif algo_id == 3: # Blowfish
                algo_name = "Blowfish"
                key = get_random_bytes(key_len) 
                cipher = Blowfish.new(key, Blowfish.MODE_CBC)

            elif algo_id == 4: # CAST5
                algo_name = "CAST5"
                final_len = min(key_len, 16)
                key = get_random_bytes(final_len)
                cipher = CAST.new(key, CAST.MODE_CBC)
            
            else:
                return None 

            # Padding
            if algo_name != "ChaCha20":
                block_size = cipher.block_size
                encrypted_data = pad(data, block_size)
            else:
                encrypted_data = data

            # Ölçüm
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_cpu = process.cpu_percent(interval=None)
            start_mem = process.memory_info().rss / (1024 * 1024)

            if algo_name == "AES" and hasattr(cipher, 'encrypt_and_digest'):
                cipher.encrypt_and_digest(encrypted_data)
            else:
                cipher.encrypt(encrypted_data)

            end_time = time.time()
            end_cpu = process.cpu_percent(interval=None)
            end_mem = process.memory_info().rss / (1024 * 1024)

            exec_time = (end_time - start_time) * 1000
            cpu_usage = (start_cpu + end_cpu) / 2
            if cpu_usage == 0: cpu_usage = 0.1 
            
            mem_usage = end_mem - start_mem
            if mem_usage < 0: mem_usage = 0

            sec_score = self.get_security_score(algo_name, len(key))

            return {
                "Algorithm": algo_name,
                "Time_ms": exec_time,
                "CPU_Percent": cpu_usage,
                "Memory_MB": mem_usage,
                "Security_Score": sec_score
            }

        except Exception as e:
            return None

    def fitness(self, x, data_size_mb=50):
        """
        Amaç fonksiyonu (Cost Function).
        Minimize edilmeye çalışılır.
        """
        # Veri Yükü Simülasyonu: 
        # Gerçekte 50 MB veri üretmek RAM'i şişirebilir, o yüzden
        # 5 MB üretip süreyi çarpanla büyüteceğiz (Projeksiyon).
        # Bu sayede hem hızlı test ederiz hem de algoritma "büyük veri" sanar.
        
        simulation_chunk = 5 # 5 MB gerçek veri
        multiplier = data_size_mb / simulation_chunk # Çarpan (Örn: 10 katı)
        
        total_bytes = simulation_chunk * 1024 * 1024
        dummy_data = get_random_bytes(total_bytes)

        result = self.run_encryption(x, dummy_data)

        if result is None:
            return self.PENALTY

        # Süreyi projeksiyonla büyüt
        real_time = result["Time_ms"] * multiplier
        
        # Normalizasyon (Gevşetilmiş sınırlar)
        norm_T = real_time / 5000.0   # 5 saniye baz alınır
        norm_CPU = result["CPU_Percent"] / 100.0 
        norm_M = abs(result["Memory_MB"]) / 50.0 
        norm_S = result["Security_Score"] / 1.0

        cost = (self.wT * norm_T) + \
               (self.wCPU * norm_CPU) + \
               (self.wM * norm_M) - \
               (self.wS * norm_S)

        return cost

if __name__ == "__main__":
    benchmark = CryptoBenchmark()
    print("--- Test Başlıyor (Motor Kontrolü) ---")
    
    # Test parametreleri: AES, GCM, 256-bit, Buffer yok, Thread yok
    params1 = [0, 1, 2, 0, 0] 
    try:
        score1 = benchmark.fitness(params1)
        print(f"AES-256 Test Skoru: {score1:.5f}")
    except Exception as e:
        print(f"Hata: {e}")