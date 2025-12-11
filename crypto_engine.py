import time
import os
import psutil
import statistics
import tracemalloc
from Crypto.Cipher import AES, ChaCha20_Poly1305, DES3, Blowfish, CAST
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

class CryptoBenchmarkV4:
    def __init__(self):
        # Ağırlıklar
        self.wT = 0.35
        self.wCPU = 0.25
        self.wM = 0.05
        self.wS = 0.35  
        self.PENALTY = 1e9 

    def get_scientific_security_score(self, algo_name, key_len_bytes, mode_name=None):
        """ NIST SP 800-57 Part 1 Rev 5 standartlarına göre puanlama. """
        bits = key_len_bytes * 8
        score = 0.0

        if algo_name == "AES":
            if bits >= 256: score = 1.0       
            elif bits >= 192: score = 0.85
            else: score = 0.70
        elif algo_name == "ChaCha20":
            score = 0.95                      
        elif algo_name in ["Twofish", "Blowfish"]:
            if bits >= 128: score = 0.50
            else: score = 0.30
        elif algo_name == "3DES":
            score = 0.20                      
        elif algo_name == "CAST5":
            score = 0.40

        if mode_name == "GCM" or algo_name == "ChaCha20":
            score += 0.05 
        elif mode_name == "ECB":
            score -= 0.50 
        
        return min(max(score, 0.0), 1.0)

    def _run_single_test(self, params, data):
        """
        Tekil test koşucusu.
        params: [AlgoID, ModID, KeyIdx, BufferSize]
        """
        algo_map = {0: "AES", 1: "ChaCha20", 2: "3DES", 3: "Blowfish", 4: "CAST5"}
        
        algo_name = algo_map.get(int(round(params[0])), "AES")
        mode_val = int(round(params[1]))
        key_idx = int(round(params[2]))
        buffer_size = int(round(params[3]))

        try:
            cipher = None
            key = b""
            mode_name = "STREAM"

            # --- Kripto Kurulum ---
            if algo_name == "AES":
                key_sizes = [16, 24, 32]
                k_len = key_sizes[min(key_idx, 2)]
                key = get_random_bytes(k_len)
                
                if mode_val == 1: # GCM
                    mode_name = "GCM"
                    nonce = get_random_bytes(12) 
                    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                elif mode_val == 2: # CTR
                    mode_name = "CTR"
                    nonce = get_random_bytes(8)
                    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
                else: # CBC
                    mode_name = "CBC"
                    iv = get_random_bytes(16)
                    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

            elif algo_name == "ChaCha20":
                mode_name = "AEAD"
                key = get_random_bytes(32)
                nonce = get_random_bytes(12)
                cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)

            elif algo_name == "3DES":
                mode_name = "CBC"
                k_len = 16 if key_idx == 0 else 24
                key = get_random_bytes(k_len)
                iv = get_random_bytes(8)
                cipher = DES3.new(key, DES3.MODE_CBC, iv=iv)

            elif algo_name == "Blowfish":
                mode_name = "CBC"
                bs_keys = [16, 32, 56]
                key = get_random_bytes(bs_keys[min(key_idx, 2)])
                iv = get_random_bytes(8)
                cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv=iv)
            
            else: # CAST5
                mode_name = "CBC"
                key = get_random_bytes(16)
                iv = get_random_bytes(8) 
                cipher = CAST.new(key, CAST.MODE_CBC, iv=iv)

            # --- DÜZELTME: Buffer Boyutunu Blok Boyutuna Hizala ---
            # Optimizer rastgele sayı üretir (örn: 12345).
            # CBC modu blok boyutunun (örn: 16) tam katını ister.
            # 12345 % 16 != 0 olduğu için hata verir.
            # Bunu engellemek için buffer_size'ı kırpıyoruz.
            
            blk_size = getattr(cipher, 'block_size', 1) # Blok boyutu (AES:16, ChaCha:1)
            
            if blk_size > 1:
                remainder = buffer_size % blk_size
                buffer_size -= remainder # Fazlalığı at
                if buffer_size == 0: buffer_size = blk_size # 0 olursa en az 1 blok yap
            
            # --------------------------------------------------------

            # Padding (Sadece CBC için tüm veri padlenir)
            data_proc = data
            if mode_name == "CBC":
                data_proc = pad(data, blk_size)

            # --- ÖLÇÜM ---
            tracemalloc.start() 
            start_cpu = time.process_time()
            start_wall = time.perf_counter()

            total_len = len(data_proc)
            
            if mode_name in ["GCM", "AEAD"]:
                # GCM/Poly1305 Chunking
                for i in range(0, total_len, buffer_size):
                    chunk = data_proc[i:i+buffer_size]
                    cipher.encrypt(chunk)
                cipher.digest() 
            
            else: # CBC, CTR
                for i in range(0, total_len, buffer_size):
                    chunk = data_proc[i:i+buffer_size]
                    # Buffer hizalandığı için artık burada hata çıkmayacak!
                    cipher.encrypt(chunk)

            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Metrikler
            wall_ms = (end_wall - start_wall) * 1000
            cpu_ms = (end_cpu - start_cpu) * 1000
            mem_mb = peak_mem / (1024 * 1024)
            
            sec_score = self.get_scientific_security_score(algo_name, len(key), mode_name)

            return {
                "Time": wall_ms,
                "CPU_Time": cpu_ms,
                "Memory": mem_mb,
                "Security": sec_score,
                "Params": f"{algo_name}-{mode_name}"
            }

        except Exception as e:
            # Hata olursa yine de görelim ama programı kırmayalım
            # print(f" [!] Hata: {str(e)}") 
            return None

    def benchmark_with_stats(self, params, data, repeats=5):
        valid_results = []
        self._run_single_test(params, data) # Warmup
        
        for _ in range(repeats):
            res = self._run_single_test(params, data)
            if res: valid_results.append(res)
        
        if not valid_results: return None
        
        return {
            "Time": statistics.median([r["Time"] for r in valid_results]),
            "CPU_Time": statistics.median([r["CPU_Time"] for r in valid_results]),
            "Memory": statistics.median([r["Memory"] for r in valid_results]),
            "Security": valid_results[0]["Security"],
            "Params": valid_results[0]["Params"]
        }