from pathlib import Path
import numpy as np
import PIL.Image as Image
import pyzapo as pz
import time

generator = pz.Sequential([
    pz.Linear(100, 512),
    pz.LeakyReLU(),
    pz.Linear(512, 2352),
    pz.Tanh()
])

weights_path = Path('weights/anime_generator_v1.npz')

if not weights_path.exists():
    raise FileNotFoundError(f'Ваги не знайдено: {weights_path}')

generator.load_weight(str(weights_path))
print('Навчені ваги завантажено!')

start_time = time.time()

noise = np.random.normal(0, 1, (1, 100)).astype(np.float32)
gen_vector = generator.forward(noise)

generation_time = time.time() - start_time

pixels = ((gen_vector + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
img_matrix = pixels.reshape(28, 28, 3)

output_filename = f'generated_anime_{int(time.time())}.png'

Image.fromarray(img_matrix, mode='RGB').save(output_filename)

print(f"Image saved: {output_filename}")
print(f"Generation time: {generation_time:.5f} seconds")
