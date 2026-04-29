import matplotlib.pyplot as plt
import pandas as pd

print("Generating Zoomed-In Fault Graphs...")

# 1. Load the perfectly cleaned data
df = pd.read_csv(r'C:\SEC project\hybrid_motor_data.csv')

# Ensure numeric types
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['amplified flux'] = pd.to_numeric(df['amplified flux'], errors='coerce')
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
df = df.dropna(subset=['time', 'amplified flux', 'speed'])
df = df.sort_values(by='time')

# Function to generate a zoomed-in dual-axis graph
def plot_fault_zoom(df_subset, fault_name, fault_start, fault_end, color_span, filename):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- LEFT Y-AXIS (Flux) ---
    ax1.plot(df_subset['time'], df_subset['amplified flux'], color='purple', linewidth=1.5, label='Amplified Flux')
    ax1.set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplified Flux', color='purple', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.grid(True)

    # --- RIGHT Y-AXIS (Speed) ---
    ax2 = ax1.twinx()
    ax2.plot(df_subset['time'], df_subset['speed'], color='teal', linewidth=1.5, label='Speed (RPM)')
    ax2.set_ylabel('Speed (RPM)', color='teal', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='teal')

    plt.title(f'{fault_name} - Flux & Speed Analysis', fontsize=14, fontweight='bold')

    # Highlight the exact fault zone
    ax1.axvspan(fault_start, fault_end, color=color_span, alpha=0.3, label=f'{fault_name} Active')

    # Combine legends neatly
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- Graph 1: Overload Fault (Zoomed between 340s to 480s) ---
print("Plotting Overload Fault...")
df_overload = df[(df['time'] >= 340) & (df['time'] <= 480)]
plot_fault_zoom(df_overload, 'Overload Fault', 353, 467, 'red', 'Zoom_Overload_Graph.png')

# --- Graph 2: Single Phase Open Fault (Zoomed between 480s to 540s) ---
print("Plotting Single Phase Open Fault...")
df_phase = df[(df['time'] >= 480) & (df['time'] <= 540)]
plot_fault_zoom(df_phase, 'Single Phase Open Fault', 493, 524, 'orange', 'Zoom_PhaseOpen_Graph.png')

print("✅ Both zoomed-in graphs saved successfully in your SEC project folder!")