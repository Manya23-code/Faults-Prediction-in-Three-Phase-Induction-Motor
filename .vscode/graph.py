import matplotlib.pyplot as plt
import pandas as pd

# Load your clean data
df = pd.read_csv('hybrid_motor_data.csv')

plt.figure(figsize=(12, 5))
# Let's plot the Amplified Flux to show the actual physical sensor readings
plt.plot(df['Timestamp'], df['Amplified_Flux'], color='purple', linewidth=1)

plt.title('Magnetic Flux vs Time ( Hardware Data)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Amplified Flux (Sensor View)')
plt.grid(True)

# Highlight where the faults happened!
plt.axvspan(353, 467, color='red', alpha=0.3, label='Overload Fault')
plt.axvspan(493, 524, color='orange', alpha=0.3, label='Phase Open Fault')

plt.legend()
plt.tight_layout()
plt.savefig('Motor_Fault_Graph.png', dpi=300)
plt.show()
print("Graph saved as 'Motor_Fault_Graph.png'!")