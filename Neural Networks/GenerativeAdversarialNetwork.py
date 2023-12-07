import numpy as np
import matplotlib.pyplot as plt

# Generator
def build_generator(input_size, output_size):
    return {
        'weights': np.random.randn(output_size, input_size),
        'bias': np.zeros((output_size, 1))
    }

def generate_fake_data(generator, num_samples):
    return np.random.randn(generator['weights'].shape[0], num_samples)

# Discriminator
def build_discriminator(input_size):
    return {
        'weights': np.random.randn(1, input_size),
        'bias': np.zeros((1, 1))
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def discriminate(discriminator, data):
    return sigmoid(np.dot(discriminator['weights'], data) + discriminator['bias'])

# Training
def train_gan(generator, discriminator, num_epochs, learning_rate):
    for epoch in range(num_epochs):
        # Generate fake data
        fake_data = generate_fake_data(generator, 100)

        # Train discriminator on real data
        real_data = np.random.randn(1, 100)
        discriminator_output_real = discriminate(discriminator, real_data)

        # Train discriminator on fake data
        discriminator_output_fake = discriminate(discriminator, fake_data)

        # Update discriminator parameters using gradient descent
        discriminator['weights'] -= learning_rate * (np.dot(discriminator_output_real - discriminator_output_fake, real_data.T) / 100)
        discriminator['bias'] -= learning_rate * np.sum(discriminator_output_real - discriminator_output_fake) / 100

        # Train generator to fool discriminator
        fake_data = generate_fake_data(generator, 100)
        discriminator_output_fake = discriminate(discriminator, fake_data)

        # Update generator parameters using gradient descent
        generator['weights'] -= learning_rate * np.dot(discriminator['weights'].T, discriminator_output_fake)
        generator['bias'] -= learning_rate * np.sum(discriminator_output_fake)

        # Print progress
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Discriminator Output Real: {discriminator_output_real.mean()}, Discriminator Output Fake: {discriminator_output_fake.mean()}')

# Generate samples from the trained generator
def generate_samples(generator, num_samples):
    return generate_fake_data(generator, num_samples)

# Create and train the GAN
input_size = 100
output_size = 1
generator = build_generator(input_size, output_size)
discriminator = build_discriminator(output_size)

# Train the GAN
train_gan(generator, discriminator, num_epochs=1000, learning_rate=0.01)

# Generate samples from the trained generator
generated_samples = generate_samples(generator, num_samples=100)

# Plot the generated samples
plt.scatter(range(100), generated_samples, label='Generated Samples')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()
