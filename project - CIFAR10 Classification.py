# %% [markdown]
# #### Executive summary
# This project presents a deep learning approach to classifying images using the CIFAR-10 dataset. The primary challenge of overfitting was addressed through batch normalization, data augmentation, and, as an extension beyond the project scope,Dropout.
# 
# * **Handling overfitting**: The CIFAR-10 dataset consists of 60,000 images from 
#   10 classes with equal distribution. To handle overfitting, we generate plausible 
#   variations of training samples by expanding the dataset through augmentation.
# * **Neural network architecture**: We implemented different CNN architectures using 
#   tanh and ReLU activations, BatchNormalization, Data Augmentation, and grayscale 
#   conversion.
# * **Conclusion**: BatchNormalization stabilized training but did not address 
#   overfitting on its own. Data augmentation mitigated overfitting, keeping the gap 
#   between training and validation loss narrow, with accuracy at 70%. Combining 
#   augmentation, BatchNormalization, and Dropout pushed accuracy to approximately 81%.
# 

# %%
import os
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns

# %%
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# %%
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

# %% [markdown]
# Let's display the architecture of your model so far:

# %%
model.summary()

# %%
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

# %% [markdown]
# Here's the complete architecture of your model:

# %%
model.summary()

# %% [markdown]
# The network summary shows that (4, 4, 64) outputs were flattened into vectors of shape (1024) before going through two Dense layers.

# %% [markdown]
# ### Compile and train the model

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs = 10, 
                    validation_data=(test_images, test_labels))

# %% [markdown]
# ### Evaluate the model

# %%
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 2)

# %%
print(test_acc)

# %% [markdown]
# This simple CNN has achieved a test accuracy of around 68%. 

# %% [markdown]
# #### Task 2
# ##### Repeat task 1 using tanh as activation

# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'tanh', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'tanh'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'tanh'))

model.summary()

# %%
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'tanh'))
model.add(layers.Dense(10))

model.summary()

# %%
model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, 
                    epochs = 10, 
                    validation_data = (test_images, test_labels))

# %%
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 2)

print(test_acc)

# %% [markdown]
# #### Task 2
# We obtain an accuracy of approximately 63% and notice that it is lower than when using ReLU. 
# 
# The computation of the ReLU activation function is simpler. "The ReLU activation function makes model training easier when using different parameter initialization methods" ([Dive into Deep Learning (2023)](#ref-d2l)).The main reason we usually do not use tanh in CNNs is that it suffers from the 'vanishing gradient problem', its gradient approaches zero when inputs are large, halting learning. Additionally, the function is 'computationally expensive' because it requires calculating an exponential term, as seen in the formula $tanh(z) = \frac{2}{1+e^{-2z}}-1$. 

# %% [markdown]
# #### Task 3 
# The CIFAR-10 data set consists of 60,000 colour pictures with 32x32 pixels. The small image size (32x32) allows for quick training and experimentation with convolutional netural networks. We start with few filters (32) to capture simple features (lines), and increase to more filters (64) deeper in the network to capture complex combinations (shapes/objects) as the spatial dimension shrinks.

# %% [markdown]
# #### Task 4 – Extend the code (ReLU) by incorporating batch normalization. 
# We start by decoupling activation and Conv2D so batch normalization (BN) can be sandwiched between them. According to [Deep learning with Python](#ref-chollet), the output of Conv2D layer gets normalized, the layer does not need its own bias vector, why we pass 'use_bias = False' as an argument.

# %%
model = models.Sequential()

# Decoupling activation and Conv2D so BN can be sandwiched between them 
# Because the output of the Conv2D layer gets normalized, the layer
# doesn't need its own bias vector.
model.add(layers.Conv2D(32, (3, 3), use_bias = False, input_shape = (32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), use_bias = False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), use_bias = False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, use_bias = False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))

# %%
import keras

model.compile(
    optimizer = keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs = 30, 
                    validation_data=(test_images, test_labels))


# %%
df_history = pd.DataFrame(history.history) # Create a DataFrame from history
df_history['epoch'] = range(1, len(df_history) + 1)

# Melt plot to fit seaborns long-format
df_plot = df_history.melt(id_vars = 'epoch', 
                          value_vars = ['loss', 'val_loss'], 
                          var_name = 'Dataset', 
                          value_name = 'Loss')

plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")
sns.lineplot(data = df_plot, 
             x = 'epoch', 
             y = 'Loss', 
             hue = 'Dataset', 
             style = 'Dataset', 
             markers = True)
plt.title("[CIFAR-10 Batch Normalization] Training and Validation Loss", 
          fontsize = 15)
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %%
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 2)

# %% [markdown]
# We observe the model's accuracy improve to 69%, with the training loss decreasing smoothly and steadily. This confirms that batch normalization has successfully stabilized the network's optimization. However, the learning curve reveals clear signs of overfitting: the validation loss fails to converge, bouncing around 0.75 and trending upward. While the model performs well on the training data, it struggles to generalize to unseen examples. To address this high variance, [Dive into Deep Learning (2023)](#ref-d2l) suggests introducing regularization techniques such as data augmentation or dropout.

# %% [markdown]
# According to [Chollet (2025)](#ref-chollet), batch normalization "can adaptively normalize data even as the mean and variance change over time during training." The main mechanism is that during training it uses the mean and variance of each batch to normalize samples — much like standard z-score standardization. Crucially, it simultaneously tracks an exponential moving average of these statistics to enable inference once training is complete.
# 
# Regarding *why* it works, Chollet and notes there are various hypotheses but "no certitudes." This uncertainty is echoed in [Dive into Deep Learning (2023)](#ref-d2l), which similarly acknowledges the lack of a definitive theoretical explanation. While Ioffe and Szegedy (2015) originally proposed that it operates by "reducing internal covariate shift," this has since been contested. Santurkar et al. (2018) argue that the primary benefit is a smoothing effect on the optimization landscape, which makes gradients more predictive and allows for faster convergence and less sensitivity to hyperparameter choices.
# 
# Finally, Chollet notes that placing normalization *before* the activation function maximizes the utilization of ReLU, which motivates our architecture above.

# %% [markdown]
# #### Task 5 Apply Data Augmentation

# %%
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1), 
    tf.keras.layers.RandomZoom(0.2),
])

plt.figure(figsize=(15, 6))

for i in range(5):
    plt.subplot(2, 5, i + 1)
    original = train_images[i]
    plt.imshow(train_images[i])

    plt.title(f"Original {i}")
    plt.axis("off")

    #Augmented images
    plt.subplot(2, 5, i + 6)
    image_batch = tf.expand_dims(original, 0)
    augmented = data_augmentation(image_batch, training = True)

    plt.imshow(augmented[0])
    plt.title(f"Augmented {i}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# %%
model = models.Sequential()

model.add(layers.Input(shape=(32, 32, 3)))
model.add(data_augmentation)

model.add(layers.Conv2D(32, (3, 3), use_bias = False))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), use_bias = False))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), use_bias = False))
model.add(layers.Activation('relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, use_bias = False))
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

# %%
history = model.fit(train_images, train_labels, epochs = 50, 
                    validation_data=(test_images, test_labels))

# %%
df_history = pd.DataFrame(history.history) # Create a DataFrame from history
df_history['epoch'] = range(1, len(df_history) + 1)

# Melt plot to fit seaborns long-format
df_plot = df_history.melt(id_vars = 'epoch', 
                          value_vars = ['loss', 'val_loss'], 
                          var_name = 'Dataset', 
                          value_name = 'Loss')

plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")
sns.lineplot(data = df_plot, x = 'epoch', y = 'Loss', hue = 'Dataset', 
             style = 'Dataset', markers = True)
plt.title("[CIFAR-10 Data Augmentation] Training and Validation Loss", 
          fontsize = 15)
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %%
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 2)

# %%
from sklearn.metrics import confusion_matrix

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis = 1)

# CIFAR-10 labels are often 2D arrays
# We flatten them to 1D to match our predictions
true_classes = test_labels.flatten()

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Create a heatmap
sns.heatmap(cm, annot = True, 
            fmt = 'd', 
            cmap = 'Blues', 
            xticklabels = class_names, yticklabels = class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# We observe an accuracy of approximately 70% using data augmentation with the same architecture as our Task 1 benchmark. While the training loss decreases smoothly, the validation loss exhibits some volatility. Both curves converge closely by epoch 20, suggesting that data augmentation has been effective at improving generalization. the model shows no clear signs of overfitting, as the gap between training and validation loss remains narrow throughout training. 
# 
# By augmenting the training data we generate plausible variations of training samples, effectively expanding the dataset. The goal, according to [Chollet (2025)](ref-#chollet), "is that at training time, your model will never see the exact same picture twice". By exposing the model to transformations such as rotation, flip, or zoom, it learns to generalize better rather than memorizing specific pixels.
# 
# To further regularize the network and stabilize convergence, [Chollet (2025)](#ref-chollet) proposes implementing Dropout. Although not required for this assignment, we will explore this in the final stage of the project.
# 
# Further analysis of the confusion matrix reveals that the model generalizes well on visually distinct classes with unique structural features (e.g., Ship or Automobile). Performance degrades on classes sharing high visual resemblance, such as Cat, Dog, and Deer, where the model struggles to discern the fine-grained features required to distinguish these animals, leading to frequent misclassifications among them.

# %% [markdown]
# #### Task 6 - Grayscale
# 

# %%
train_gray = tf.image.rgb_to_grayscale(train_images)
test_gray = tf.image.rgb_to_grayscale(test_images)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

history = model.fit(train_gray, train_labels, epochs = 20, 
                    validation_data=(test_gray, test_labels))


# %%
df_history = pd.DataFrame(history.history) # Create a DataFrame from history
df_history['epoch'] = range(1, len(df_history) + 1)

# Melt plot to fit seaborns long-format
df_plot = df_history.melt(id_vars = 'epoch', 
                          value_vars = ['loss', 'val_loss'], 
                          var_name = 'Dataset', 
                          value_name = 'Loss')

plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")
sns.lineplot(data = df_plot, x = 'epoch', y = 'Loss', hue = 'Dataset', 
             style = 'Dataset', markers = True)
plt.title("[CIFAR-10 Grayscale] Training and Validation Loss", 
          fontsize = 15)
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %%
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_gray,  test_labels, verbose = 2)

# %% [markdown]
# As anticipated, overfitting manifests early, appearing within the first 15 epochs in the absence of batch normalization or data augmentation. Furthermore, the overall accuracy is lower (60%) compared to the baseline RGB architecture from Task 1. This performance drop is expected, as the model is forced to rely exclusively on structural features (edges, textures, and shapes). We hypothesize that color is a strong feature in CIFAR-10 as certain classes exhibit strong color correlations (e.g., frogs are typically green, deer are brown), which the grayscale model cannot utilize.

# %% [markdown]
# #### Appendix: Task 5 - Further Investigation

# %%
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Build model
model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))

# Augmentation
model.add(data_augmentation)

# Block 1
model.add(layers.Conv2D(32, (3, 3), padding = 'same', use_bias = False)) # Added padding
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Block 2
model.add(layers.Conv2D(64, (3, 3), padding= 'same', use_bias = False)) # Added padding
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2)) # Added dropout

# Block 3
model.add(layers.Conv2D(64, (3, 3), padding = 'same', use_bias = False)) # Added padding
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3)) # Increased droput

# Classification Head
model.add(layers.Flatten())
model.add(layers.Dense(64, use_bias = False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10))

model.compile(
    optimizer = keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs = 300, 
                    validation_data=(test_images, test_labels))

# %%
df_history = pd.DataFrame(history.history) # Create a DataFrame from history
df_history['epoch'] = range(1, len(df_history) + 1)

# Melt plot to fit seaborns long-format
df_plot = df_history.melt(id_vars = 'epoch', 
                          value_vars = ['loss', 'val_loss'], 
                          var_name = 'Dataset', 
                          value_name = 'Loss')

plt.figure(figsize = (10, 6))
sns.set_style("darkgrid")
sns.lineplot(data = df_plot, x = 'epoch', y = 'Loss', hue = 'Dataset', 
             style = 'Dataset', markers = True)
plt.title("[CIFAR-10 RBG - BN & Dropout] Training and Validation Loss", 
          fontsize = 15)
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %%
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose = 2)

# %%
from sklearn.metrics import confusion_matrix

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis = 1)

# CIFAR-10 labels are often 2D arrays
# We flatten them to 1D to match our predictions
true_classes = test_labels.flatten()

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Create a heatmap
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', 
            xticklabels = class_names, yticklabels = class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# We observe a clear improvement in stability so a combination of Droput and data augmentation yields in a notably stable learning curve that effectively suppresses overfitting. Despite the extended training, the model plateaus at approximately 80% accuracy, suggesting the architecture has reached its representational capacity and cannot extract deeper features from the data. 
# 
# It should be noted that no hyperparameter tuning has been applied, and adjusting the learning rate could yield incremental gains. To meaningfully push accuracy higher, the next logical step would be a more complex architecture or transfer learning with pre-trained ImageNet weights.

# %% [markdown]
# ### References
# * <a name="ref-santurkar"> </a>**Santurkar et al.(2018)**, "How does Batch Normalization Help Optimization?", Gradient Science, 2018. [Read Article](https://gradientscience.org/batchnorm/)
# 
# * <a name="ref-d2l"> </a> **Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023)**, Dive into Deep Learning. Cambridge University Press. Available at: [https://d2l.ai](https://d2l.ai)
# 
# * <a name="ref-keras"></a> **Keras API documentation**, Available via: [keras.io](https://keras.io)
# 
# * <a name="ref-chollet"></a> **Chollet, F. (2025)**, Deep Learning with Python (3rd ed.). Manning Publications. Online-version available via: [deeplearningwithpython.io](https://deeplearningwithpython.io)
# 
# * <a name="ref-ibm"></a> **IBM Research (2023)**, What is data leakage in machine learning? IBM Think Topics. Available via: [ibm.com](https://www.ibm.com/think/topics/data-leakage-machine-learning)
# 
# ### Acknowledgements
# * **Foundational Coding:** The architectural design and implementation of the Keras model followed best practices and principles outlined in [Deep Learning with Python by François Chollet](https://deeplearningwithpython.io), ensuring a robust and standardized approach of deep learning.
# * **Technical Implementation:** The logic for tensor dimensionality manipulation within the data augmentation loop was developed with the assistance of an AI tool. 
# 
# ### Technical Remarks: 
# The experimental results indicate a performance plateau at approximately 70% accuracy when utilizing the standard techniques outlined in the project scope (Augmentation and BN). However, by extending the architecture with Dropout and optimizing model capacity, we successfully elevated the accuracy to 80-81%. To surpass this threshold, future work would likely benefit from Transfer Learning to leverage robust feature maps learned from larger datasets. Additionally, a more exhaustive hyperparameter search could optimize the network. 


