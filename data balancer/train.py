import pandas as pd
import time
import sys

nSample = 1000

# # Fungsi untuk animasi loading
# def loading_animation(message="Loading", duration=5):
#     chars = "|/-\\"
#     end_time = time.time() + duration
#     while time.time() < end_time:
#         for char in chars:
#             sys.stdout.write(f'\r{message} {char}')
#             sys.stdout.flush()
#             time.sleep(0.1)
#     sys.stdout.write('\rDone!            \n')  # Menghapus animasi saat selesai

# # Tampilkan animasi loading sebelum proses dimulai
# loading_animation("Processing data")

# Baca file CSV
data = pd.read_csv(r'../Dataset UTS/emnist-bymerge-train.csv')

# Pastikan kolom label ada, ganti 'label' sesuai dengan nama kolom dalam dataset Anda jika berbeda
label_column = '24'

# Hitung jumlah kelas dan tentukan jumlah sampel per kelas yang diinginkan
num_classes = data[label_column].nunique()
samples_per_class = nSample // num_classes  # Sampel merata per kelas

# Ambil sampel per kelas
balanced_data = data.groupby(label_column, group_keys=False).apply(lambda x: x.sample(samples_per_class))

# Pastikan jumlah total sampel adalah 1000
if len(balanced_data) > nSample:
    balanced_data = balanced_data.sample(nSample)

# Tampilkan distribusi kelas untuk verifikasi
print(balanced_data[label_column].value_counts())

# Simpan hasilnya ke file CSV baru jika diperlukan
balanced_data.to_csv('balanced_emnist_subset.csv', index=False)

print("Data has been processed and saved successfully.")