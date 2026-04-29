import matplotlib.pyplot as plt
import pandas as pd

print("Generating the Dual-Axis Continuous Data Graph...")

# 1. Load the data
df = pd.read_csv(r'C:\SEC project\hybrid_motor_data.csv')

# 2. Safety Check: Convert to numbers using CORRECT column names
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['amplified flux'] = pd.to_numeric(df['amplified flux'], errors='coerce')
df['speed'] = pd.to_numeric(df['speed'], errors='coerce')

# Drop any rows where any of these three sensors missed a reading
df = df.dropna(subset=['time', 'amplified flux', 'speed'])

# 3. Sort strictly by time so the line flows forward
df = df.sort_values(by='time') 

# 4. Draw the Graph (Dual Y-Axis)
fig, ax1 = plt.subplots(figsize=(12, 6))

# --- LEFT Y-AXIS (For Amplified Flux) ---
ax1.plot(df['time'], df['amplified flux'], color='purple', linewidth=1, label='Amplified Flux')
ax1.set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Amplified Flux (Sensor View)', color='purple', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='purple')
ax1.grid(True)

# --- RIGHT Y-AXIS (For Speed) ---
ax2 = ax1.twinx()  # This creates the second Y-axis sharing the same X-axis
ax2.plot(df['time'], df['speed'], color='teal', linewidth=1, alpha=0.7, label='Motor Speed (RPM)')
ax2.set_ylabel('Speed (RPM)', color='teal', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='teal')

plt.title('Motor Physics: Magnetic Flux & Speed vs Time', fontsize=16, fontweight='bold')

# Highlight Fault Zones (Attached to ax1)
ax1.axvspan(353, 467, color='red', alpha=0.2, label='Overload Fault')
ax1.axvspan(493, 524, color='orange', alpha=0.2, label='Phase Open Fault')

# Combine legends from both axes into one box
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.savefig('Motor_Flux_Speed_Graph.png', dpi=300)
plt.show()

print("Graph saved successfully!")