import os

tflite_path = r"C:\SEC project\motor_model.tflite"
header_path = r"C:\SEC project\model_data.h"

def file_to_header(tflite_path, header_path):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    with open(header_path, 'w') as f:
        f.write('#ifndef MOTOR_MODEL_DATA_H\n')
        f.write('#define MOTOR_MODEL_DATA_H\n\n')
        f.write(f'unsigned char motor_model_tflite[] = {{\n  ')
        
        # Convert binary data to hex format for C++
        for i, byte in enumerate(data):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:
                f.write('\n  ')
        
        f.write('\n};\n')
        f.write(f'unsigned int motor_model_tflite_len = {len(data)};\n\n')
        f.write('#endif\n')

if os.path.exists(tflite_path):
    file_to_header(tflite_path, header_path)
    print(f"✅ SUCCESS: C++ Header file created at {header_path}")
else:
    print("❌ ERROR: motor_model.tflite not found!")