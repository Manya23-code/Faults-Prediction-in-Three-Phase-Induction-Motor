import tensorflow as tf
import os

# 1. Configuration
h5_model_path = 'motor_brain.h5'
tflite_model_path = 'motor_brain.tflite'
header_file_path = 'motor_brain.h'

def create_esp32_header():
    # --- STEP 1: Check if the Brain exists ---
    if not os.path.exists(h5_model_path):
        print(f"❌ Error: {h5_model_path} not found! You MUST run 'python train.py' first.")
        return

    # --- STEP 2: Convert H5 to TFLite ---
    print("🔄 Converting H5 to TFLite format...")
    try:
        model = tf.keras.models.load_model(h5_model_path, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Created: {tflite_model_path}")
    except Exception as e:
        print(f"❌ TFLite Conversion Failed: {e}")
        return

    # --- STEP 3: Convert TFLite to C Header (.h) ---
    print("🔄 Generating C Header for ESP32...")
    with open(tflite_model_path, 'rb') as f:
        data = f.read()

    with open(header_file_path, 'w') as f:
        f.write('#include <stdint.h>\n\n')
        f.write('const unsigned char motor_model_data[] = {\n  ')
        for i, byte in enumerate(data):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        f.write(f'const int motor_model_data_len = {len(data)};\n')
    
    print(f"✅ Created: {header_file_path} ({len(data)} bytes)")
    print("\n🚀 SUCCESS! You can now use motor_brain.h in your ESP32 project.")

if __name__ == "__main__":
    create_esp32_header()