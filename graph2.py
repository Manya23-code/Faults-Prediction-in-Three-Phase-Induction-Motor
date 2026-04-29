import pandas as pd
import matplotlib.pyplot as plt
import os

print("Generating the Direct Correlation Mapping: Flux vs Speed Cluster Plot...")

# Define the absolute path to your perfectly generated CSV from Turn 10
file_path = r"C:\SEC project\hybrid_motor_data.csv"

# Double check file existence again just to be bulletproof
if not os.path.exists(file_path):
    print(f"❌ ERROR: Cleaned data file not found at {file_path}. Did Step 1 finish successfully?")
else:
    # Load and clean data (Same cleaning steps as training)
    df = pd.read_csv(file_path)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['amplified flux'] = pd.to_numeric(df['amplified flux'], errors='coerce')
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['time', 'amplified flux', 'speed'])

    # === SEGMENT DATA BASED ON YOUR ESTABLISHED TIMESTAMPS ===
    # Using the same precise ranges from Turn 5 / Training Script

    # 1. Healthy Operating State (No Load + Normal Load background contrast)
    df_healthy = df[(df['time'] >= 190.0) & (df['time'] < 350.0)]

    # 2. OVERLOAD FAULT (Emphasized with Red Markers)
    df_overload = df[(df['time'] >= 353.0) & (df['time'] <= 468.0)]

    # 3. SINGLE PHASE OPEN FAULT (Emphasized with Orange Markers)
    # Merging ranges from Turn 5 for direct correlation view simplicity
    df_phaseopen = df[(df['time'] >= 493.0) & (df['time'] <= 505.99) | (df['time'] >= 516.0) & (df['time'] <= 524.99)]

    print(f"✅ Segments loaded: Healthy({len(df_healthy)} points), Overload({len(df_overload)}), PhaseOpen({len(df_phaseopen)})")

    # === Create the Scatter Plot ===
    plt.figure(figsize=(10, 8))

    # Plot Background Healthy points (low visibility to emphasize faults)
    plt.scatter(df_healthy['speed'], df_healthy['amplified flux'], color='skyblue', label='Healthy State (No Load/Normal)', alpha=0.3, s=15, marker='.')

    # Plot Overload Points (Emphasized Red Circles)
    plt.scatter(df_overload['speed'], df_overload['amplified flux'], color='red', label='Overload Fault Cluster', alpha=0.8, s=25, marker='o', edgecolors='black')

    # Plot Phase Open Points (Emphasized Orange Triangles)
    plt.scatter(df_phaseopen['speed'], df_phaseopen['amplified flux'], color='orange', label='Phase Open Fault Cluster', alpha=0.8, s=25, marker='^', edgecolors='darkorange')

    # === Finalize Graph Aesthetics ===
    plt.title('Motor Operation Mapping: Amplified Flux vs Speed', fontsize=16, fontweight='bold')
    plt.xlabel('Motor Speed (RPM)', fontsize=12, fontweight='bold')
    plt.ylabel('Amplified Flux (Sensor View)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', shadow=True)

    # Save and show
    plt.tight_layout()
    output_filename = r"C:\SEC project\Flux_vs_Speed_Cluster_Mapping.png"
    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"✅ Mapping graph successfully saved as: {output_filename}")