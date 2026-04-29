import pandas as pd
import matplotlib.pyplot as plt
import os

print("Generating Direct Flux vs Speed Scatter Plots...")

# 1. Load the cleaned data
file_path = r"C:\SEC project\hybrid_motor_data.csv"
df = pd.read_csv(file_path)

df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['amplified flux'] = pd.to_numeric(df['amplified flux'], errors='coerce')
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
df = df.dropna(subset=['time', 'amplified flux', 'speed'])

# 2. Extract exactly the Fault Zones
df_overload = df[(df['time'] >= 353.0) & (df['time'] <= 468.0)]
df_phaseopen = df[(df['time'] >= 493.0) & (df['time'] <= 524.99)]

# Function to plot Flux vs Speed
def plot_flux_vs_speed(df_subset, fault_name, point_color, filename):
    plt.figure(figsize=(8, 6))
    
    # X-axis is Speed, Y-axis is Flux
    plt.scatter(df_subset['speed'], df_subset['amplified flux'], 
                color=point_color, alpha=0.7, edgecolors='black', s=40)
    
    plt.title(f'{fault_name}: Amplified Flux vs Speed', fontsize=15, fontweight='bold')
    plt.xlabel('Motor Speed (RPM)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplified Flux (Sensor View)', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- Graph 1: Overload Fault ---
print("Plotting Overload Fault (Flux vs Speed)...")
plot_flux_vs_speed(df_overload, 'Overload Fault', 'red', 'Flux_vs_Speed_Overload.png')

# --- Graph 2: Single Phase Open Fault ---
print("Plotting Single Phase Open Fault (Flux vs Speed)...")
plot_flux_vs_speed(df_phaseopen, 'Single Phase Open Fault', 'orange', 'Flux_vs_Speed_PhaseOpen.png')

print("✅ Both Flux vs Speed graphs saved successfully!")