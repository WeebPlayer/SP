import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Definisikan Universe Variabel
# Input (Antecedents)
pm25 = ctrl.Antecedent(np.arange(0, 501, 1), 'PM2.5')
aqi = ctrl.Antecedent(np.arange(0, 501, 1), 'AQI')

# Output (Consequent)
kualitas = ctrl.Consequent(np.arange(0, 101, 1), 'Kualitas Udara')

# 2. Definisikan Fungsi Keanggotaan (Membership Functions)

# PM2.5 (µg/m³)
pm25['Baik'] = fuzz.trimf(pm25.universe, [0, 0, 50])
pm25['Sedang'] = fuzz.trimf(pm25.universe, [25, 50, 150])
pm25['Tidak Sehat'] = fuzz.trimf(pm25.universe, [50, 150, 250])
pm25['Berbahaya'] = fuzz.trimf(pm25.universe, [150, 500, 500])

# AQI
aqi['Baik'] = fuzz.trimf(aqi.universe, [0, 0, 50])
aqi['Sedang'] = fuzz.trimf(aqi.universe, [25, 75, 125])
aqi['Tidak Sehat'] = fuzz.trimf(aqi.universe, [100, 150, 200])
aqi['Berbahaya'] = fuzz.trimf(aqi.universe, [175, 500, 500])

# Kualitas Udara (0-100)
kualitas['Sangat Baik'] = fuzz.trimf(kualitas.universe, [0, 0, 25])
kualitas['Baik'] = fuzz.trimf(kualitas.universe, [15, 35, 55])
kualitas['Sedang'] = fuzz.trimf(kualitas.universe, [45, 60, 75])
kualitas['Buruk'] = fuzz.trimf(kualitas.universe, [65, 80, 90])
kualitas['Sangat Buruk'] = fuzz.trimf(kualitas.universe, [85, 100, 100])

# 3. Definisikan Aturan Fuzzy (Rules)
# Kita tambahkan 8 aturan logis selain 2 aturan wajib Anda.

rule1 = ctrl.Rule(pm25['Baik'] & aqi['Baik'], kualitas['Sangat Baik'])
rule2 = ctrl.Rule(pm25['Berbahaya'] | aqi['Berbahaya'], kualitas['Sangat Buruk']) # Aturan OR

rule3 = ctrl.Rule(pm25['Sedang'] & aqi['Baik'], kualitas['Baik'])
rule4 = ctrl.Rule(pm25['Baik'] & aqi['Sedang'], kualitas['Baik'])
rule5 = ctrl.Rule(pm25['Sedang'] & aqi['Sedang'], kualitas['Sedang'])
rule6 = ctrl.Rule(pm25['Tidak Sehat'] & aqi['Baik'], kualitas['Sedang'])
rule7 = ctrl.Rule(pm25['Baik'] & aqi['Tidak Sehat'], kualitas['Sedang'])
rule8 = ctrl.Rule(pm25['Tidak Sehat'] & aqi['Sedang'], kualitas['Buruk'])
rule9 = ctrl.Rule(pm25['Sedang'] & aqi['Tidak Sehat'], kualitas['Buruk'])
rule10 = ctrl.Rule(pm25['Tidak Sehat'] & aqi['Tidak Sehat'], kualitas['Buruk'])

# 4. Buat Sistem Kontrol (Inference System)
kualitas_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, 
                                     rule6, rule7, rule8, rule9, rule10])
prediktor = ctrl.ControlSystemSimulation(kualitas_ctrl)

# 5. Uji Coba Sistem (Test Cases)
print("--- 🧪 Uji Coba Sistem Prediksi Kualitas Udara ---")

# Test Case 1: Keduanya Baik
prediktor.input['PM2.5'] = 10
prediktor.input['AQI'] = 20
prediktor.compute()
print(f"PM2.5: 10, AQI: 20  -> Kualitas Udara: {prediktor.output['Kualitas Udara']:.2f}")

# Test Case 2: Keduanya Sedang
prediktor.input['PM2.5'] = 75
prediktor.input['AQI'] = 80
prediktor.compute()
print(f"PM2.5: 75, AQI: 80  -> Kualitas Udara: {prediktor.output['Kualitas Udara']:.2f}")

# Test Case 3: Keduanya Tidak Sehat
prediktor.input['PM2.5'] = 200
prediktor.input['AQI'] = 180
prediktor.compute()
print(f"PM2.5: 200, AQI: 180 -> Kualitas Udara: {prediktor.output['Kualitas Udara']:.2f}")

# Test Case 4: Campuran (Baik & Tidak Sehat)
prediktor.input['PM2.5'] = 30
prediktor.input['AQI'] = 190
prediktor.compute()
print(f"PM2.5: 30, AQI: 190 -> Kualitas Udara: {prediktor.output['Kualitas Udara']:.2f}")

# Test Case 5: Pemicu Aturan 'Berbahaya' (OR)
prediktor.input['PM2.5'] = 80  # (Sedang)
prediktor.input['AQI'] = 450  # (Berbahaya)
prediktor.compute()
print(f"PM2.5: 80, AQI: 450 -> Kualitas Udara: {prediktor.output['Kualitas Udara']:.2f}")

print("--------------------------------------------------")

# 6. Untuk Visualisasi (dijalankan di environment lokal)
# pm25.view()
# aqi.view()
# kualitas.view()
# Untuk melihat hasil defuzzifikasi spesifik (contoh Test Case 2):
# prediktor.input['PM2.5'] = 75
# prediktor.input['AQI'] = 80
# prediktor.compute()
# kualitas.view(sim=prediktor)
