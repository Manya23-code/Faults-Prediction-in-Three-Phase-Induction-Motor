import matplotlib.pyplot as plt
import pandas as pd

print("Generating the Continuous Data Graph...")

# 1. Load the data
df = pd.read_csv('hybrid_motor_data.csv')

# 2. Safety Check: Convert to numbers and drop any fully corrupt rows
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
df['Flux'] = pd.to_numeric(df['Flux'], errors='coerce')
df = df.dropna(subset=['Timestamp', 'Flux'])

# 3. Sort strictly by time so the line flows forward and bridges the gap automatically
df = df.sort_values(by='Timestamp') 

# 4. Draw the Graph (No line breaks this time!)
plt.figure(figsize=(12, 5))
plt.plot(df['Timestamp'], df['Flux'], color='purple', linewidth=1)

plt.title('Magnetic Flux vs Time (Hardware Data)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Flux (Sensor View)')
plt.grid(True)

# Highlight Fault Zones
plt.axvspan(353, 467, color='red', alpha=0.3, label='Overload Fault')
plt.axvspan(493, 524, color='orange', alpha=0.3, label='Phase Open Fault')

plt.legend()
plt.tight_layout()
plt.savefig('Motor_Fault_Graph.png', dpi=300)
plt.show()

print("Graph saved successfully!")