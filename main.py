from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, Add, GlobalAveragePooling2D, Dense, Input, Conv2DTranspose, Reshape, Activation, Flatten, Embedding, Concatenate, AveragePooling2D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from generator import generator_network
from discriminator import discriminator_network
from classifier import classifier
import wandb
import os




generator_optimizer = Adam(learning_rate=0.00005, beta_1=0, beta_2=0.999)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0, beta_2=0.999)
classifier_optimizer = Adam(learning_rate=0.0002, beta_1=0, beta_2=0.999)

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def calculate_classifier_loss(true_labels, predicted_labels):
  loss = tf.keras.losses.categorical_crossentropy(true_labels, predicted_labels)
  return loss

def calculate_generator_loss(fake_output):
  loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)

  return loss

def calculate_discriminator_loss(real_output, fake_output):
  real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)

  return real_loss + fake_loss


num_classes = 10
z_dim = 128


@tf.function
def train_step(images, labels):
    batch_size = images.shape[0]

    # Generate noise vector and random class embeddings
    noise = tf.random.normal([batch_size, z_dim])
    random_labels = tf.random.uniform([batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    class_embeddings = tf.one_hot(random_labels, depth=num_classes)
    print(class_embeddings, random_labels)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as class_tape:
        # Train the generator
        print("Training generator")
        generated_images = generator([noise, class_embeddings], training=True)

        # Train the discriminator
        print("Training discriminator with real input")
        real_output = discriminator([images, labels], training=True)

        print("Training the discriminator with fake input")
        fake_output = discriminator([generated_images, class_embeddings], training=True)

        # Classifier outputs for real and fake images
        print("Training classifier")
        predicted_labels = classifier_network(images, training = True)
        #fake_class_output = classifier(generated_images)

        # Calculate losses
        gen_loss = calculate_generator_loss(fake_output)
        disc_loss = calculate_discriminator_loss(real_output, fake_output)
        class_loss = calculate_classifier_loss(labels, predicted_labels)

    print("Calculating gradients")
    # Calculate gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    class_gradients = class_tape.gradient(class_loss, classifier_network.trainable_variables)

    print("applying gradients")
    # Apply gradients
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    classifier_optimizer.apply_gradients(zip(class_gradients, classifier_network.trainable_variables))
    
    return gen_loss, disc_loss, class_loss



if __name__ == '__main__':
    
    #config

    NUM_EPOCHS = 10
    BUFFER_SIZE = 60000
    BATCH_SIZE = 2
    SELECTED_TRAIN_IMGS_SIZE = 10000

    wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb.init(
      # Set the project where this run will be logged
      project="Triple-BiGGAN", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"Server-Run",
      # Track hyperparameters and run metadata
      config={
      "architecture": "Triple-BiGGAN",
      "dataset": "CIFAR-100",
      "epochs": NUM_EPOCHS,
      })
    
    # Load the dataset
    (train_imgs, train_labels), (test_imgs, test_labels) = cifar10.load_data()
    wandb.log({"train_imgs": train_imgs, "train_labels": train_labels, "test_imgs": test_imgs, "test_labels": test_labels})

    np.random.seed(42)

    # Generate random indices
    random_indices = np.random.choice(len(train_imgs), size=SELECTED_TRAIN_IMGS_SIZE, replace=False)

    # Select random images and labels
    random_train_imgs = train_imgs[random_indices]
    random_train_labels = train_labels[random_indices]

    # Resize images
    resized_images = tf.image.resize(random_train_imgs, [256, 256])

    #normalize resized images
    normalized_resized_images = resized_images / 255.0
    test_images = test_imgs / 255.0

    num_classes = 10
    train_labels = tf.keras.utils.to_categorical(random_train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # Create a TensorFlow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((resized_images, train_labels)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels))
    
    generator = generator_network()
    discriminator = discriminator_network()
    classifier_network = classifier()

    classifier_network.compile(optimizer=classifier_optimizer,)
    generator.compile(optimizer=generator_optimizer, )
    discriminator.compile(optimizer=discriminator_optimizer,)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        gen_train_loss = []
        disc_train_loss = []
        class_train_loss = []
        
        
        start = time.time()
        for real_images, real_labels in train_dataset:
            generator_loss, discriminator_loss, classifier_loss = train_step(real_images, real_labels)
            
            #print(np.mean(generator_loss), np.mean(discriminator_loss), classifier_loss)
            
            gen_train_loss.append(float(generator_loss))
            disc_train_loss.append(float(discriminator_loss))
            
            for loss in classifier_loss:
                class_train_loss.append(float(loss))

            #print(f"Epoch {epoch+1}/{num_epochs}")
            #print()
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        wandb.log({'epoch': epoch,
              'gen_loss': np.mean(gen_train_loss),
              'disc_loss': np.mean(disc_train_loss),
                'class_loss': np.mean(class_train_loss)
                })
      

    # Save the trained models
    generator.save_weights('generator_model.h5')
    discriminator.save_weights('discriminator_model.h5')
    classifier_network.save_weights('classifier_model.h5')

    wandb.finish()

