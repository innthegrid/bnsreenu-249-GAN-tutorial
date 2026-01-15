from numpy import asarray
from numpy.random import randn, randint
from keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt

from training import generate_latent_points

# Load model
model = load_model('cifar_conditional_generator_model.h5')

# Generate multiple images
latent_points, labels = generate_latent_points(100, 100)
# Specify labels - generate 10 sets of labesl each going for 0 to 9
labels = asarray([x for _ in range(10) for x in range(10)])
# Generate images
X = model.predict([latent_points, labels])
# Scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)

# Plot the result
def show_plot(examples, n):
  for i in range(n * n):
    plt.subplot(n, n, 1 + i)
    plt.axis('off')
    plt.imshow(examples[i, :, :, :])
  plt.show()

show_plot(X, 10)