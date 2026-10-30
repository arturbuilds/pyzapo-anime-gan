import pyzapo as pz
import numpy as np
import PIL.Image as Image

lr = 0.0002
batch_size = 64

generator = pz.Sequential([
    pz.Linear(100, 512),
    pz.LeakyReLU(),
    pz.Linear(512, 2352),
    pz.Tanh()
])

descriminator = pz.Sequential([
    pz.Linear(2352, 128),
    pz.LeakyReLU(),
    pz.Linear(128, 1),
    pz.Sigmoid()
])

loss_fn = pz.BCELoss()

dataset = np.load('anime_images.npy')
dataset = dataset.reshape(-1, 2352) 
num_samples = dataset.shape[0]

dataset = np.array(dataset, dtype=np.float32)

indices = np.arange(num_samples)

for epoch in range(500):
    np.random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        real_images = dataset[batch_indices]
        current_batch = real_images.shape[0]

        if current_batch < batch_size:
            continue

        pred_real = descriminator.forward(real_images)
        loss_real = loss_fn.forward(pred_real, np.ones((current_batch, 1)))

        descriminator.backward(loss_fn.backward())
        descriminator.step(lr)

        noise = np.random.normal(0, 1, (current_batch, 100))
        fake_images = generator.forward(noise)

        pred_fake = descriminator.forward(fake_images)
        loss_fake = loss_fn.forward(pred_fake, np.zeros((current_batch, 1)))

        descriminator.backward(loss_fn.backward())
        descriminator.step(lr)

        noise = np.random.normal(0, 1, (current_batch, 100))
        fake_images = generator.forward(noise)

        pred_fake = descriminator.forward(fake_images)
        loss_G = loss_fn.forward(pred_fake, np.ones((current_batch, 1)))

        grad_from_loss = loss_fn.backward()
        grad_through_D = descriminator.backward(grad_from_loss)

        generator.backward(grad_through_D)
        generator.step(lr)

    if epoch % 1 == 0:
        print(f'Епоха: {epoch:4d} | Loss D: {loss_real + loss_fake:.4f} | Loss G: {loss_G:.4f}')

        test_noise = np.random.normal(0, 1, (1, 100))
        gen_vector = generator.forward(test_noise)

        pixels = ((gen_vector + 1.0) * 127.5).astype(np.uint8)
        
        img_matrix = pixels.reshape(28, 28, 3)

        img = Image.fromarray(img_matrix, mode='RGB')
        img.save(f'images/anime_ready_epoch_{epoch}.png')

print('\nНавчання завершено!')

generator.save_weight('weights/anime_generator_v1.npz')
print('Фінальні ваги збережені!')
