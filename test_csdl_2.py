import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

name = ''
folder = ''
file = f'{name}_rasp'
output_dir = f'{folder}/{file}'
images_directory = f'{folder}/{name}'
os.makedirs(output_dir, exist_ok = True)

image_files = [filename for filename in os.listdir(images_directory) if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')]
image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

for i, filename in enumerate(image_files):
    img_path = os.path.join(images_directory, filename)
    frame = cv2.imread(img_path)
    input_image_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_image_resized = np.expand_dims(input_image_resized, axis=0)
    input_image_resized = input_image_resized.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_image_resized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    file_base_name = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f'{file_base_name}.npy')
    np.save(output_file, output_data.flatten())  
    print(f'Extracted features from: {filename}')   
print('All feature vectors are saved')