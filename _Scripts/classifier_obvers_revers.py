import os
import re
import shutil
import math
import time
import cv2
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras import layers
from keras.activations import relu,linear
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.regularizers import L1, L2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import optuna
import logging
#logging.getLogger("tensorflow").setLevel(logging.ERROR)
#tf.autograph.set_verbosity(0)

########################################################################################################################
# Functions Section
########################################################################################################################


def clean_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except:
            pass


def display_2_images(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1)
    axes[0].set_title(f'Original Image\n{img1.shape}')
    axes[0].axis('off')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'Modified Image\n{img2.shape}')
    axes[1].axis('off')
    plt.show()


def save_model_hyperparameters(model, history, file_path):
    with open(file_path, 'w+') as file:
        # Save model architecture
        file.write("Model Architecture:\n")
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        # Save hyperparameters
        file.write("\nHyperparameters:\n")
        for layer in model.layers:
            if isinstance(layer, Dense):
                file.write(f"Layer: {layer.name}\n")
                file.write(f"  Units: {layer.units}\n")
                file.write(f"  Activation: {layer.activation.__name__}\n")
                file.write(f"  Regularization L1: {layer.get_config().get('kernel_regularizer', {}).get('config', {}).get('l1', 0.0)}\n")
                file.write(f"  Regularization L2: {layer.get_config().get('kernel_regularizer', {}).get('config', {}).get('l2', 0.0)}\n")

        # Save optimizer details (assuming the model uses a single optimizer)
        optimizer = model.optimizer
        file.write("\nOptimizer:\n")
        file.write(f"  Name: {optimizer.get_config()['name']}\n")
        file.write(f"  Epochs: {len(history.history['loss'])}\n")
        file.write(f"  Learning Rate: {optimizer.get_config().get('learning_rate', 0.0)}\n")


def save_model_hyperparameters_to_csv(model, history, csv_file):
    # Extract model architecture details
    model_architecture = []
    model.summary(print_fn=lambda x: model_architecture.append(x.strip()))

    # Extract hyperparameters
    hyperparameters = []
    hyperparameters.append("Layer,Units,Activation,Regularization L1,Regularization L2")
    for layer in model.layers:
        if isinstance(layer, Dense):
            hyperparameters.append(f"{layer.name},{layer.units},{layer.activation.__name__},{layer.get_config().get('kernel_regularizer', {}).get('config', {}).get('l1', 0.0)},{layer.get_config().get('kernel_regularizer', {}).get('config', {}).get('l2', 0.0)}")

    # Extract optimizer details
    optimizer = model.optimizer
    optimizer_details = [f"Optimizer,{optimizer.get_config()['name']},{optimizer.get_config().get('learning_rate', 0.0)}"]

    # Extract number of epochs from the training history
    num_epochs = [f"Number of Epochs,{len(history.history['loss'])}"]

    # Write to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Architecture"])
        writer.writerows(map(lambda x: [x], model_architecture))
        writer.writerow(["Hyperparameters"])
        writer.writerows(map(lambda x: x.split(','), hyperparameters))
        writer.writerow(["Optimizer"])
        writer.writerows(map(lambda x: x.split(','), optimizer_details))
        writer.writerow(["Training History"])
        writer.writerows(map(lambda x: x.split(','), num_epochs))


def save_img_nparray(image, filename):
    image = image/255
    image_array = np.array(image)
    #flattened_image = image_array.flatten()
    filename = filename.split('.')[0] + '.npy'
    np.save(filename, image_array)


def square_img(img_path, image_size):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    circle_center_x, circle_center_y, circle_r = int(width / 2), int(height / 2), int(min(height, width) / 2 * 0.9)
    edge_offset = 10
    x1 = int(max(circle_center_x, circle_r) - min(circle_center_x, circle_r)) + edge_offset
    x2 = int(max(circle_center_x, circle_r) + min(circle_center_x, circle_r)) - edge_offset
    y1 = int(max(circle_center_y, circle_r) - min(circle_center_y, circle_r)) + edge_offset
    y2 = int(max(circle_center_y, circle_r) + min(circle_center_y, circle_r)) - edge_offset
    cropped_image = gray_image[y1:y2, x1:x2]
    try:
        square_image = cv2.resize(cropped_image, (image_size, image_size))
    except:
        display_2_images(gray_image, cropped_image)
        pass
    #display_2_images(cropped_image, square_image)
    return square_image


def img_prep(img_input_dir, img_proc_dir, npy_dir, year_1, year_2, image_size=120):
    clean_folder(img_proc_dir)
    clean_folder(npy_dir)
    count = 0
    time_start = time.time()
    years_to_save = [y for y in range(year_1, year_2 + 1)]
    print(f"Years to save: {years_to_save}\n")
    for filename in os.listdir(img_input_dir):
        try:
            year = int(filename.split('_')[1][:4])
        except:
            year = int(filename.split('_')[2][:4])
        if year not in years_to_save:
       # if int(filename.split('_')[2][:4]) not in years_to_save:
            #pass
            continue
        else:
            count += 1
            image_path = os.path.join(img_input_dir, filename)
            square_image = square_img(image_path, image_size)
            cv2.imwrite(img_proc_dir + 'GS_' + filename, square_image)
            save_img_nparray(square_image, npy_dir + 'GS_' + filename)
        #if count == 10: break

    delta_time = time.time() - time_start
    print(f'\nScript took {delta_time:.3f} sec')
    npy_concat(npy_dir, '_Dataset', '_y')


def npy_concat(npy_dir, output_filename_img, output_filename_y):
    for file in [f'{output_filename_img}.npy', f'{output_filename_y}.npy']:
        print(file)
        file_path = os.path.join(npy_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    count = 0
    flattened_images = []
    num_files = len(os.listdir(npy_dir))
    y_vector = np.zeros((num_files, 1), dtype=int)
    print(y_vector.shape)
    for filename in os.listdir(npy_dir):
        loaded_array = np.load(npy_dir + filename)
        #print(f'{count}  {filename} {loaded_array.shape}')
        flattened_image = loaded_array.flatten()
        flattened_images.append(flattened_image)
        y_vector[count] = int(filename.split('.')[0][-1]) - 1
        count += 1
        os.remove(os.path.join(npy_dir, filename))

    concatenated_images = np.array(flattened_images)
    np.save(npy_dir + output_filename_img, concatenated_images)
    np.save(npy_dir + output_filename_y, y_vector)
    print(f'\n{concatenated_images.shape}\n'
          f'{count} images have been concatenated into {output_filename_img}.npy')
    print(f'{y_vector.shape}\n'
          f'{count} outputs have been saved into {output_filename_y}.npy\n')


def classify_obverse_revers(img_input_dir):
    global ml_models_dir
    loaded_model = tf.keras.models.load_model(f'{ml_models_dir}classifier_obverse_revers.h5')
    total_coins = 0
    to_flip = 0
    for img_name in os.listdir(img_input_dir):
        image_path = os.path.join(img_input_dir, img_name)
        #print(f'{count}  {img_name}')
        square_image = square_img(image_path)
        x_length = np.square(square_image.shape[0])
        X_check = np.zeros((1, x_length), dtype=int)
        y_check = np.zeros((1, 1), dtype=int)
        #print(X_check.shape, y_check.shape)
        #print(X_check, y_check)
        #cv2.imwrite(img_proc_dir + 'GS_' + img_name, square_image)
        X_check[0] = np.array(square_image/255).flatten()
        y_check[0] = int(img_name.split('.')[0][-1]) - 1
        #print(X_check.shape, y_check.shape)
        #print(X_check, y_check)
        prediction = loaded_model.predict(X_check[0].reshape(1, x_length), verbose=0)
        print(img_name, prediction)
        if prediction >= 0.5:
            # if prediction[0][1] > prediction[0][0]:
            yhat = 1
        else:
            yhat = 0
        if y_check[0, 0] != yhat:
            to_flip += 1
            print(f'{img_name} to flip')
            flip_dir = img_input_dir + 'to_flip/'
            if not os.path.exists(flip_dir):
                os.makedirs(flip_dir)
            side = int(img_name.split('.')[0][-1])
            if side == 1:
                file2_path = os.path.join(img_input_dir, img_name.split('.')[0][:-1] + '2.' + img_name.split('.')[1])
            else:
                file2_path = os.path.join(img_input_dir, img_name.split('.')[0][:-1] + '1.' + img_name.split('.')[1])
            shutil.copy2(image_path, flip_dir)
            shutil.copy2(file2_path, flip_dir)
            #switch_sides(img_input_dir, img_name)
        total_coins += 1
    #if to_flip > 0:
    #    switch_sides(flip_dir)
    print(f'Found {to_flip}({total_coins}) coins flipped')


def train_model(X, y):
    global ml_models_dir
    model = Sequential([tf.keras.Input(shape=(X.shape[1],)),  # specify input size
            #Dense(units=256, activation='relu', kernel_regularizer=L2 (0.001)),
            #Dropout(0.33),
            #Dense(units=256, activation='relu', kernel_regularizer=L2 (0.003)),
            #Dropout(0.33),
            Dense(units=128, activation='relu', kernel_regularizer=L2 (0.003)),
            Dropout(0.33),
            Dense(units=32, activation='relu', kernel_regularizer=L2 (0.003)),
            #Dropout(0.16),
            #Dense(units=2, activation='linear', kernel_regularizer=L2 (0.005)),
            Dense(units=1, activation='sigmoid'),
        ], name="my_model")
    model.compile(
        #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.00002),
        metrics=['accuracy'])

    time_start = time.time()
    history = model.fit(X, y, epochs=200, verbose=2, validation_split=0.20)
    delta_time = time.time() - time_start
    print(f'\nTraining took {delta_time:.3f} sec')
    test_loss, test_accuracy = model.evaluate(X, y)
    #print('\nFinal loss: ' + str(history.history['loss'][-1]) + '\n')
    print(f"\nTest Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")
    print(f"CV Loss: {history.history['val_loss'][-1]:.3f}, CV Accuracy: {history.history['val_accuracy'][-1]:.3f}")
    plot_loss(history)
    #save_model_hyperparameters(model, history, f'{ml_models_dir}model_hyperparameters.txt')
    #save_model_hyperparameters_to_csv(model, history, f'{ml_models_dir}model_hyperparameters.csv')
    model.save(f'{ml_models_dir}classifier_obverse_revers.h5')
    return model


def train_model_2(X, y, md_id):
    global ml_models_dir
    learning_rate = md_id['learning_rate']
    val_split = md_id['val_split']
    num_layers = md_id['num_layers']
    num_epochs = md_id['num_epochs']
    num_units_1 = md_id['num_units_1']
    num_units_2 = md_id['num_units_2']
    num_units_3 = md_id['num_units_3']
    activation_1 = md_id['activation_1']
    activation_2 = md_id['activation_2']
    activation_3 = md_id['activation_3']
    regularization_1 = md_id['regularization_1']
    regularization_2 = md_id['regularization_2']
    regularization_3 = md_id['regularization_3']
    dropout_1 = md_id['dropout_1']
    dropout_2 = md_id['dropout_2']
    dropout_3 = md_id['dropout_3']
    input_shape = X.shape[1]
    num_samples, img_size = X.shape
    model = Sequential()
    input_layer = Input(shape=(input_shape,))
    model.add(input_layer)
    print(num_units_1, activation_1, regularization_1, dropout_1)
    print(num_units_2, activation_2, regularization_2, dropout_2)
    layer_1 = Dense(units=num_units_1, activation=activation_1, kernel_regularizer=L2 (regularization_1))
    model.add(layer_1)
    layer_drop_1 = Dropout(dropout_1)
    model.add(layer_drop_1)
    layer_2 = Dense(units=num_units_2, activation=activation_2, kernel_regularizer=L2 (regularization_2))
    model.add(layer_2)
    layer_drop_2 = Dropout(dropout_2)
    model.add(layer_drop_2)
    if num_units_3 != 0 and activation_3 != 0:
        layer_3 = Dense(units=num_units_3, activation=activation_3, kernel_regularizer=L2 (regularization_3))
        model.add(layer_3)
        layer_drop_3 = Dropout(dropout_3)
        model.add(layer_drop_3)
    output_layer = Dense(units=1, activation='sigmoid')
    model.add(output_layer)
    #print(model.summary())
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy'])

    time_start = time.time()
    history = model.fit(X, y, epochs=num_epochs, verbose=2, validation_split=val_split)
    delta_time = time.time() - time_start
    print(f'\nTraining took {delta_time:.3f} sec')
    test_loss, test_accuracy = model.evaluate(X, y)
    print(f"\nTest Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")
    print(f"CV Loss: {history.history['val_loss'][-1]:.3f}, CV Accuracy: {history.history['val_accuracy'][-1]:.3f}")
    plot_loss(history, md_id, num_samples, int(np.sqrt(img_size)), delta_time)
    model.save(f'{ml_models_dir}classifier_obverse_revers.h5')
    return model


def predict_test(model_name, X, y, plot = 0):
    m, n = X.shape
    print(m, n)
    failed = []
    global ml_models_dir
    loaded_model = tf.keras.models.load_model(f'{ml_models_dir}{model_name}')

    for i in range(m):
        prediction = loaded_model.predict(X[i].reshape(1, n), verbose=0)
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0
        if y[i, 0] != yhat:
            failed.append(i)
    print(f'{len(failed)} failed out of {m}\n', failed)
    print(f"\nAccuracy = {(1 - len(failed) / m) * 100:.2f}%")

    if plot != 0:
        fig, axes = plt.subplots(3, 4, figsize=(8, 4))
        for i, ax in enumerate(axes.flat):
            print(i, ax)
            failed_index = failed[i]
            X_random_reshaped = X[failed_index].reshape((int(np.sqrt(n)), int(np.sqrt(n))))#.T
            ax.imshow(X_random_reshaped, cmap='gray')
            ax.set_title(f"{failed_index}: {y[failed_index, 0]+1}", fontsize=10, color='red', fontweight='bold')
            ax.set_title(f"{y[failed_index, 0]+1}")
            ax.set_axis_off()
        plt.show()


def visualize_coins(X, y, items, predict=0, plot = 0):
    m, n = X.shape
    fig, axes = plt.subplots(int(4), int(m/4), figsize=(8, 4))
    fails = 0
    #random_indices = np.random.permutation(items**2)
    random_indices = np.random.permutation(m)
    #print(len(y), len(random_indices))
    #print(random_indices)
    for i, ax in enumerate(axes.flat):
        #random_index = np.random.randint(m)
        random_index = random_indices[i]
        X_random_reshaped = X[random_index].reshape((int(np.sqrt(n)), int(np.sqrt(n))))#.T
        ax.imshow(X_random_reshaped, cmap='gray')
        if predict == 1:
            global model
            prediction = model.predict(X[random_index].reshape(1, n), verbose=0)
            #print(random_index, prediction)
            if prediction >= 0.5:
            #if prediction[0][1] > prediction[0][0]:
                yhat = 1
            else:
                yhat = 0
            if y[random_index, 0] != yhat:
                fails += 1
                ax.set_title(f"{random_index}: {y[random_index, 0]+1} - {yhat+1}", fontsize=10, color='red', fontweight='bold')
            else:
                ax.set_title(f"{random_index}: {y[random_index, 0]+1} - {yhat+1}", fontsize=10, color='blue')
        else:
            ax.set_title(f"{y[random_index, 0]+1}")
        ax.set_axis_off()

    print(f"\nAccuracy = {(1 - fails / items / items) * 100:.2f}%")
    if plot != 0:
        plt.show()


def largest_num_in_filename(ml_models_dir, ext):
    # Get all files in the folder with '.png' extension
    png_files = [file for file in os.listdir(ml_models_dir) if file.lower().endswith(f'.{ext}')]
    # Filter files with names ending with a number
    numbered_files = [file for file in png_files if re.search(r'\d+$', file.split('.')[0])]
    numbers = [int(re.search(r'\d+', file).group()) for file in numbered_files]
    return max(numbers, default=0)


def plot_loss(history, hyperparams, num_samples, img_size, calc_time):
    global ml_models_dir
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(loss))
    tick_step = max(1, int(len(loss) / 10))  # Ensuring a minimum step of 1

    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 3])

    # Subplot 1: Training and Validation Loss
    ax1 = plt.subplot(gs[0])
    ax1.plot(epochs_range, loss, label='Training Loss')
    ax1.plot(epochs_range, val_loss, label='Validation Loss')
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.xticks(np.arange(0, len(loss), tick_step), np.arange(0, len(loss), tick_step).astype(int))
    ax1.legend(loc='upper right')
    ax1.set_title('Training and Validation Loss')

    # Subplot 2: Training and Validation Accuracy
    ax2 = plt.subplot(gs[1])
    ax2.plot(epochs_range, acc, label='Training Accuracy')
    ax2.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xticks(np.arange(0, len(loss), tick_step), np.arange(0, len(loss), tick_step).astype(int))
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    ax2.legend(loc='lower right')
    ax2.set_title('Training and Validation Accuracy')

    # Adjust layout to leave space at the bottom for additional text
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Adding annotation for hyperparameters
    val_split = hyperparams['val_split']
    learning_rate = hyperparams['learning_rate']
    num_layers = hyperparams['num_layers']
    num_epochs = hyperparams['num_epochs']
    num_units_1 = hyperparams['num_units_1']
    num_units_2 = hyperparams['num_units_2']
    num_units_3 = hyperparams['num_units_3']
    activation_1 = hyperparams['activation_1']
    activation_2 = hyperparams['activation_2']
    activation_3 = hyperparams['activation_3']
    regularization_1 = hyperparams['regularization_1']
    regularization_2 = hyperparams['regularization_2']
    regularization_3 = hyperparams['regularization_3']
    dropout_1 = hyperparams['dropout_1']
    dropout_2 = hyperparams['dropout_2']
    dropout_3 = hyperparams['dropout_3']
    text_to_plot = (f'Image size: {img_size}x{img_size} pixels\n'
                    f'Dataset: {num_samples}, Training: {int(num_samples*(1-val_split))}, Val_split: {val_split}\n'
                    f'Layer_1: UN={num_units_1}, A={activation_1}, RG={regularization_1}, DP={dropout_1}\n'
                    f'Layer_2: UN={num_units_2}, A={activation_2}, RG={regularization_2}, DP={dropout_2}\n'
                    f'Epochs: {num_epochs}   Learning Rate: {learning_rate:.5f}\n')
    text_to_plot_2 = (f'Training took {int(calc_time)} s\n'
                      f'Test Accuracy: {(100*np.mean(acc[-10:])):.2f}%\n'
                      f'CV Accuracy: {(100*np.mean(val_acc[-10:])):.2f}%')
    plt.text(0.06, 0.2, text_to_plot,
             horizontalalignment='left',
             verticalalignment='top',
             transform=fig.transFigure)
    plt.text(0.95, 0.2, text_to_plot_2,
             horizontalalignment='right',
             verticalalignment='top',
             transform=fig.transFigure)
    plot_number = largest_num_in_filename(ml_models_dir, 'png')
    plt.savefig(f'{ml_models_dir}Train_hist_{plot_number+1}.png')
    #plt.show()


def switch_sides(input_folder):
    for filename in os.listdir(input_folder):
        side = int(filename.split('.')[0][-1])
        if side == 1:
            temp_file_name = "temp_switch_file"
            file1_path  = os.path.join(input_folder, filename)
            # Rename file1 to the temporary name
            os.rename(file1_path, os.path.join(input_folder, temp_file_name))
            # Rename file2 to file1's original name
            file2_path = os.path.join(input_folder, filename.split('.')[0][:-1] + '2.' + filename.split('.')[1])
            os.rename(file2_path, file1_path)
            # Rename the temporary file to file2's original name
            os.rename(os.path.join(input_folder, temp_file_name), file2_path)


def read_csv(settings_file):
    df = pd.read_csv(settings_file, index_col='ID')
    variables = list(df.columns)
    variable_values = {}
    for index, row in df.iterrows():
        variable_values[index] = {variable: row[variable] for variable in variables}
    return variable_values


def model_study(model_settings):
    global img_input_dir
    global img_aug_dir
    global img_proc_dir
    global npy_dir
    for id, md in model_settings.items():
        print(md['new_dataset'])
        if md['new_dataset'] != 0:
            image_size = md['image_size']
            aug_batch = md['aug_batch']
            data_augmentation_2(img_input_dir, img_aug_dir, batch_per_img=aug_batch, aug_originals=0)
            img_prep(img_aug_dir, img_proc_dir, npy_dir, 1700, 1799, image_size=image_size)
            X_train, X_test, y_train, y_test = dataset_prep(test_size=0.1)
        train_model_2(X_train, y_train, md)


def data_augmentation_2(img_dir, augmented_dir, batch_per_img=10, aug_originals=2):
    clean_folder(augmented_dir)
    # Create an ImageDataGenerator instance with desired augmentations
    datagen = ImageDataGenerator(
        rotation_range=15, #10,
        width_shift_range=0.05, #0.025,
        height_shift_range=0.05, #0.025,
        shear_range=0.05, #0.025,
        zoom_range=0.05, #0.025,
        brightness_range=[0.01, 1.23],
        fill_mode='nearest'
        #horizontal_flip = True,
        #vertical_flip = True
        )

    total_orig = 0
    for filename in os.listdir(img_dir):
        total_orig += 1
        img = image.load_img(os.path.join(img_dir, filename))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        i = 0
        for batch in datagen.flow(x, batch_size=1, shuffle=True, save_to_dir=augmented_dir, save_prefix=filename, save_format='png'):
            i += 1
            if i >= batch_per_img:
                break
        if total_orig >= aug_originals and aug_originals != 0:
            break
        shutil.copy2(os.path.join(img_dir, filename), augmented_dir)

    for filename in os.listdir(augmented_dir):
        new_filename = filename.split('.')[1].split('_')[-1] + '_' + filename.split('.')[0] + '.' + filename.split('.')[1].split('_')[0]
        os.rename(os.path.join(augmented_dir, filename), os.path.join(augmented_dir, new_filename))


def dataset_prep(test_size):
    X = np.load(npy_dir + "_Dataset.npy")
    y = np.load(npy_dir + "_y.npy")
    print(f'Original dataset: {X.shape}, {y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    #items_to_plot = int(math.floor(np.sqrt(len(y_test))))
    print(f'Train dataset: {X_train.shape}, {y_train.shape}')
    print(f'Test dataset: {X_test.shape}, {y_test.shape}')
    #print(f'Items to plot: {items_to_plot}')

    return X_train, X_test, y_train, y_test

########################################################################################################################
# Set the folder path
img_input_dir = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_Raw2/'
img_aug_dir = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/aug2/'
img_proc_dir = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_Grey_Squared2/'
npy_dir = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_NPY2/'
ml_models_dir = 'D:/_PROJECTS/Coins_Classification/ML_models/'
dir_to_classify = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/_to_classify/'
settings_file = 'D:/_PROJECTS/Coins_Classification/ML_models/model_settings.csv'
for dir in [img_input_dir, img_proc_dir, npy_dir, ml_models_dir, dir_to_classify]:
    if not os.path.exists(dir):
        os.makedirs(dir)


model_settings = read_csv(settings_file)

# Data augmentation
#X, y = data_augmentation(X, y, repeat = 100)
data_augmentation_2(img_input_dir, img_aug_dir, batch_per_img=5, aug_originals=0)

# Prepare images for Dataset
img_prep(img_aug_dir, img_proc_dir, npy_dir, 1700, 1799, image_size=120)

# Train the model
#model = train_model(X_train, y_train)

#model_study(model_settings)

# Verification and Vusialization
#X_train, X_test, y_train, y_test = dataset_prep(test_size=0.02)
#predict_test('classifier_obverse_revers.h5', X_test, y_test, plot = 1)

#visualize_coins(X_test, y_test, items_to_plot, predict = 1, plot = 0)

# Classify Obverse-Reverse using pre-trained model
#classify_obverse_revers(dir_to_classify)


