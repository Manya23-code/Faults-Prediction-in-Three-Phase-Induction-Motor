import tensorflow as tf
import os

h5_model_path = r"C:\SEC project\best_attention_motor_model.h5"
tflite_model_path = r"C:\SEC project\motor_model.tflite"

print("1. Loading your LSTM model...")
model = tf.keras.models.load_model(h5_model_path)

print("2. Starting Conversion with LSTM & Concrete Function fix...")

# This creates a 'concrete function' with a fixed batch size of 1
# It tells TFLite exactly what to expect: 1 sample of 15 time-steps with 2 features
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 15, 2], model.inputs[0].dtype)
)

# Convert from the concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)

# These are the magic lines to handle the LSTM complexity
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ SUCCESS! Your ESP32-ready model is saved at: {tflite_model_path}")
    print(f"TFLite Size: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")
except Exception as e:
    print(f"❌ Conversion failed: {e}")