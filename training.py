# Conditional GAN generates images using a random latent vector and corresponding label as input
# Supply a label during training so the latent vector can be associated with a specific label
# Makes the generation of images predictable

from numpy import zeros, ones
from numpy.random import randn, randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate
from matplotlib import pyplot as plt

### LOAD & PLOT DATASET ###
# CIFAR10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

(trainX, trainy), (testX, testy) = load_data()

# Plot 25 images
for i in range(25):
  plt.subplot(5, 5, 1 + i)
  plt.axis('off')
  plt.imshow(trainX[i])
plt.show()

### DEFINE DISCRIMINATOR ###
# Define the standalone discriminator model
# Given an input image, the Discriminator outputs the likelihood of the image being real
# Binary classification - true (1) or false (0) - use sigmoid activation
# Unlike regular GAN, we also provide number of classes as input
# Input to the model will be both images and labels

# Input image is size 32x32x3 - 32x32 pixels, 3 color channels (RGB)
# CIFAR10 has 10 classes
def define_discriminator(in_shape=(32,32,3), n_classes=10):
  ## LABEL INPUT ##
  # Turn the label into the right format so the model can learn

  # Create space for one label per image - shape 1
  # Input() defines the starting point of a neural network, and the format of the input
  in_label = Input(shape=(1,))

  # Embedding layer - turns positive integers into dense vectors of fixed size
  # Maps each value in the input array to a vector of a defined size
  # Weights in this layer are learned during the training process

  # Each label will be represented by a vector of size 50, which is learnt by the discriminator
  li = Embedding(n_classes, 50)(in_label)

  # Problem: The image and label are different shapes
  # n_nodes calculate how many nodes are needed (height x width)
  # 32x32 = 1024
  n_nodes = in_shape[0] * in_shape[1]
  # Scale up the label to image dimensions with linear activation
  # Shape = 1, 1023
  li = Dense(n_nodes)(li)

  # Reshape the list of 1024 numbers into 32x32 square
  # 32x32x1
  li = Reshape((in_shape[0], in_shape[1], 1))(li)

  ## IMAGE INPUT ##
  # 32x32x3
  in_image = Input(shape=in_shape)

  ## COMBINE ##
  # Concatenate label as a channel
  # 32x32x4 (4 channels, 3 for image and 1 for labels, all 32x32)
  merge = Concatenate()([in_image, li])

  # Downsample - Same as unconditional GAN
  # Combine input label with input image and supply as inputs to the model

  # fe stands for Feature Extraction
  # 128 filters, (3,3) kernel size
  # 16x16x128 (32/2 = 16)
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
  fe = LeakyReLU(alpha=0.2)(fe)

  # Downsample
  # 8x8x128
  fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
  fe = LeakyReLU(alpha=0.2)(fe)

  # Flatten feature map into shape of 8192 (8x8x128)
  fe = Flatten()(fe)

  # Dropout
  fe = Dropout(0.4)(fe)

  # Output
  # Shape = 1, range 0-1
  out_layer = Dense(1, activation='sigmoid')(fe)

  # Define model
  # Combine input label with input image and supply as inputs to the model
  model = Model([in_image, in_label], out_layer)

  # Compile model
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

test_discr = define_discriminator()
print(test_discr.summary())

### DEFINE GENERATOR ###
# Define the standalone generator model
# Takes latent vector and label as inputs
def define_generator(latent_dim, n_classes=10):
  # Input of dimension 1
  in_label = Input(shape=(1,))

  # Embedding for categorical input
  # Each label will be represented by a vector of size 50
  # Shape 1, 50
  li = Embedding(n_classes, 50)(in_label)

  # Linear multiplication
  # To match the dimensions for concatenation later
  n_nodes = 8 * 8
  # Shape 1, 64
  li = Dense(n_nodes)(li)
  # Reshape to additional channel
  li = Reshape((8, 8, 1))(li)

  # Input of dimension 100
  in_lat = Input(shape=(latent_dim,))

  # Foundation for 8x8 image
  # We will reshape input latent vector into 8x8 image as a starting point
  # So n_nodes for the Dense laer can be 128x8x8
  # So when we reshape the output, it would be 8x8x128
  # Can be slowly upscaled to 32x32 image for output
  # While defining model inputs we will combine input label and the latent input
  n_nodes = 128 * 8 * 8

  # Shape = 8192
  gen = Dense(n_nodes)(in_lat)
  gen = LeakyReLU(alpha=0.2)(gen)
  # Shape = 8x8x128
  gen = Reshape((8, 8, 128))(gen)

  # Merge image gen and label input
  # Shape = 8x8x129 (extra channel for label)
  merge = Concatenate()([gen, li])

  # Upsample to 16x16x128
  gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
  gen = LeakyReLU(alpha=0.2)(gen)

  # Upsample to 32x32x128
  gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
  gen = LeakyReLU(alpha=0.2)(gen)

  # Output - 32x32x3
  out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen)

  # Define model
  model = Model([in_lat, in_label], out_layer)

  # Model is not compiled as it is not directly trained like the discriminator

  return model

test_gen = define_generator(100, n_classes=10)
print(test_gen.summary())

### DEFINE GAN ###
# Generator is trained via GAN combined model
# Define the combined generator and discriminator model, for updating the generator
# Discriminator is trained separately, keep the discriminator constant here
def define_gan(g_model, d_model):
  d_model.trainable = False

  # Connect the Generator and Discriminator

  # First, get noise and label inputs from generator model
  # Latent vector size and label size
  gen_noise, gen_label = g_model.input

  # Get image output from the generator model
  # 32x32x3
  gen_output = g_model.output

  # Generator image output and corresponding input label are inputs to discriminator
  gan_output = d_model([gen_output, gen_label])

  # Define gan model as taking noise and label and outputting a classification
  model = Model([gen_noise, gen_label], gan_output)

  # Compile model
  opt = Adam(learning_rate=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)

  return model

# Load CIFAR images
def load_real_samples():
  # Load dataset
  (trainX, trainy), (_, _) = load_data()

  # Convert to float
  X = trainX.astype('float32')

  # Generator uses tanh activation so we have to rescale to match the output of generator
  # Scale from [0,255] to [-1,1]
  X = (X - 127.5) / 127.5

  return [X, trainy]

# Pick a batch of random real samples to train the GAN
# Train GAN on a half batch of real images and another half batch of fake images
# For each real image, we assign a label 1
# For each fake image, we assign a label 0
def generate_real_samples(dataset, n_samples):
  # Split into images and labels
  images, labels = dataset

  # Choose random instances
  ix = randint(0, images.shape[0], n_samples)

  # Seleect images and labels
  X, labels = images[ix], labels[ix]

  # Generate class labels and assign to y
  # Label=1 indicating they are real
  y = ones((n_samples, 1))

  return [X, labels], y

# Generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
  # Generate points in the latent space
  x_input = randn(latent_dim * n_samples)

  # Reshape into a batch of inputs for the network
  z_input = x_input.reshape(n_samples, latent_dim)

  # Generate labels
  labels = randint(0, n_classes, n_samples)

  return [z_input, labels]

# Use the generator to generate n fake examples, with class labels
# Input latent_dim and number of samples
def generate_fake_samples(generator, latent_dim, n_samples):
  # Generate points in latent space
  z_input, labels_input = generate_latent_points(latent_dim, n_samples)

  # Predict outputs
  images = generator.predict([z_input, labels_input])
  
  # Create class labels
  y = zeros((n_samples, 1))

  return [images, labels_input], y

### DEFINE TRAIN ###
# Train the generator and discriminator
# We look through a number of epochs to train the Discriminator
# First select a random batch of images from our true/real dataset
# Then generate a set of images using the Generator
# Feed both sets of images into the Discriminator
# Finally, set the loss parameters for both the real and fake images, and the combined loss
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
  bat_per_epo = int(dataset[0].shape[0] / n_batch)
  half_batch = int(n_batch / 2)

  for i in range(n_epochs):
    for j in range(bat_per_epo):
      # Train the discriminator on half_batch of real images and half batch of fake_images
      # Get randomly selected real samples
      [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
      # Update discriminator model weights
      d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)

      # Generate fake examples
      [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)

      # Prepare points in latent space as input for the generator
      [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)

      # The generator wants to trick the discriminator into believing the generated image is true (1)
      # Create inverted labels for the fake samples
      y_gan = ones((n_batch, 1))

      # Generator is part of combined model (directly linked with the discriminator)
      # Train the generator with latent_dim as x and 1 as y
      # Update the generator via the discriminator's error
      g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
      
      # Print losses on this batch
      print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' %(i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))

  # Save the generator model
  g_model.save('cifar_conditional_generator_model.h5')

### CREATE AND TRAIN MODELS ###
if __name__ == "__main__":
  # Size of the latent space
  latent_dim = 100

  # Create the discriminator
  d_model = define_discriminator()

  # Create the generator
  g_model = define_generator(latent_dim)

  # Create the GAN
  gan_model = define_gan(g_model, d_model)

  # Load image data
  dataset = load_real_samples()

  # Train model
  train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2)