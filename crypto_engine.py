import time
import os
import psutil
import tracemalloc  # RAM için hassas ölçüm
import numpy as np
from Crypto.Cipher import AES, ChaCha20, DES3, Blowfish, CAST
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

class CryptoBenchmarkV2:
    def __init__(self):
        # Ağırlıklar
        self.wT = 0.4
        self.wCPU = 0.3
        self.wM = 0.1
        self.wS = 0.2
        self.PENALTY = 1e9 # Çok yüksek ceza

    def get_scientific_security_score(self, algo_name, key_len_bytes, mode_name=None):
        """
        NIST ve Akademik standartlara (Effective Security Bits) göre puanlama.
        Referans: NIST SP 800-57 Part 1
        """
        bits = key_len_bytes * 8
        score = 0.0

        # 1. Temel Bit Gücü (NIST Eşdeğeri)
        if algo_name == "AES":
            if bits >= 256: score = 1.0       # AES-256 (Quantum Resistant-ish)
            elif bits >= 192: score = 0.85
            else: score = 0.70                # AES-128 (Standart Güvenli)
        
        elif algo_name == "ChaCha20":
            score = 0.90                      # Modern, güvenli (256-bit)
        
        elif algo_name == "Twofish" or algo_name == "Blowfish":
            # Blowfish 64-bit blok boyutu yüzünden doğum günü saldırılarına açıktır.
            # Anahtar uzun olsa bile blok boyutu cezası alır.
            if bits >= 256: score = 0.60
            elif bits >= 128: score = 0.50
            else: score = 0.30
            
        elif algo_name == "3DES":
            # 3DES efektif güvenliği 112-bittir (Meet-in-the-middle).
            # Ayrıca 2023 itibariyle NIST tarafından "Disallowed" statüsündedir.
            score = 0.20 
            
        elif algo_name == "CAST5":
            score = 0.40

        # 2. Mod Güvenliği (Authenticated Encryption Bonusu)
        if mode_name == "GCM" or algo_name == "ChaCha20": # ChaCha20 Poly1305 gibidir
            score += 0.10 # Bütünlük koruması (Integrity) olduğu için
        elif mode_name == "ECB":
            score -= 0.50 # ASLA KULLANILMAMALI
        
        return min(max(score, 0.0), 1.0) # 0-1 arasına sıkıştır

    def run_benchmark(self, params, data):
        """
        params: [AlgoID, ModID, KeyIdx, BlockSize, Threads]
        """
        # --- 1. Parametre Mapping (Kesin Dönüşüm) ---
        algo_map = {0: "AES", 1: "ChaCha20", 2: "3DES", 3: "Blowfish", 4: "CAST5"}
        algo_name = algo_map.get(int(round(params[0])), "AES")
        
        # Mod: 0:CBC, 1:GCM (Sadece AES), 2:CTR
        mode_val = int(round(params[1]))
        
        # Key: 0:Min, 1:Mid, 2:Max
        key_idx = int(round(params[2]))
        
        # Block Size: Buffer okuma boyutu (Cipher Block Size DEĞİL)
        # 1KB ile 64KB arası
        buffer_size = int(round(params[3]))
        
        # Thread Simülasyonu (Python GIL yüzünden gerçek thread kriptoda zordur,
        # ancak multiprocessing veya chunking simüle edilir. Biz burada tek thread 
        # saf performansa odaklanacağız, Thread parametresini "Chunking Strategy" olarak kullanacağız).
        
        try:
            cipher = None
            key = b""
            mode_name = "STREAM" # Varsayılan

            # --- 2. Algoritma Kurulumu (Kritik Düzeltmeler: IV/Nonce) ---
            if algo_name == "AES":
                key_sizes = [16, 24, 32]
                k_len = key_sizes[min(key_idx, 2)]
                key = get_random_bytes(k_len)
                
                if mode_val == 1: # GCM
                    mode_name = "GCM"
                    cipher = AES.new(key, AES.MODE_GCM) # Nonce otomatik üretilir
                elif mode_val == 2: # CTR
                    mode_name = "CTR"
                    cipher = AES.new(key, AES.MODE_CTR) # Nonce otomatik
                else: # CBC (Default)
                    mode_name = "CBC"
                    iv = get_random_bytes(AES.block_size) # KRİTİK: IV EKLENDİ
                    cipher = AES.new(key, AES.MODE_CBC, iv)

            elif algo_name == "ChaCha20":
                mode_name = "STREAM"
                key = get_random_bytes(32)
                nonce = get_random_bytes(12) # KRİTİK: Nonce 12 byte (veya 8)
                cipher = ChaCha20.new(key=key, nonce=nonce)

            elif algo_name == "3DES":
                mode_name = "CBC"
                k_len = 16 if key_idx == 0 else 24
                key = get_random_bytes(k_len)
                iv = get_random_bytes(DES3.block_size)
                cipher = DES3.new(key, DES3.MODE_CBC, iv)

            elif algo_name == "Blowfish":
                mode_name = "CBC"
                # Blowfish key: 4-56 bytes. Biz 16, 32, 56 seçelim
                bs_keys = [16, 32, 56]
                key = get_random_bytes(bs_keys[min(key_idx, 2)])
                iv = get_random_bytes(Blowfish.block_size)
                cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)
            
            else: # CAST5
                mode_name = "CBC"
                key = get_random_bytes(16)
                iv = get_random_bytes(CAST.block_size)
                cipher = CAST.new(key, CAST.MODE_CBC, iv)

            # --- 3. Padding (Blok Şifreler İçin Zorunlu) ---
            # Stream cipher (ChaCha, CTR, GCM) padding istemez.
            # CBC ister.
            data_to_encrypt = data
            if mode_name == "CBC":
                # Algoritmanın blok boyutunu dinamik al
                # PyCryptodome cipher objesinde .block_size olmayabilir, sınıftan alırız
                blk = getattr(cipher, 'block_size', 8) 
                data_to_encrypt = pad(data, blk)

            # --- 4. ÖLÇÜM BAŞLIYOR (Hassas Yöntem) ---
            
            # Bellek İzleme Başlat
            tracemalloc.start()
            
            # CPU Zamanı (İşletim sistemi gürültüsünden arınmış)
            start_cpu_time = time.process_time() 
            start_wall_time = time.time()

            # --- ŞİFRELEME ---
            # Tek seferde değil, buffer_size (x[3]) ile chunk chunk yapalım
            # Bu sayede RAM parametresinin etkisi gerçekçi olur.
            ciphertext = b""
            if mode_name == "GCM":
                ciphertext, tag = cipher.encrypt_and_digest(data_to_encrypt)
            else:
                # Chunking simülasyonu
                for i in range(0, len(data_to_encrypt), buffer_size):
                    chunk = data_to_encrypt[i:i+buffer_size]
                    cipher.encrypt(chunk)

            # Ölçüm Bitiş
            end_wall_time = time.time()
            end_cpu_time = time.process_time()
            current, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # --- 5. Metrik Hesaplama ---
            exec_time_ms = (end_wall_time - start_wall_time) * 1000
            cpu_usage_ms = (end_cpu_time - start_cpu_time) * 1000 # CPU'nun harcadığı saf süre
            memory_used_mb = peak_mem / (1024 * 1024) # Peak memory en doğru ölçümdür

            security_score = self.get_scientific_security_score(algo_name, len(key), mode_name)

            return {
                "Time": exec_time_ms,
                "CPU_Time": cpu_usage_ms,
                "Memory": memory_used_mb,
                "Security": security_score,
                "Params": f"{algo_name}-{mode_name}"
            }

        except Exception as e:
            # print(f"Hata: {e}") # Debug için açılabilir
            return None