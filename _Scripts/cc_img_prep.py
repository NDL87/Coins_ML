import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import datetime
from datetime import datetime
import time

#img_path = 'D:/_PROJECTS/Coins_Classification/img/poltina/Полтина_1832_506_67_2.png'
img_path = 'D:/_PROJECTS/Coins_Classification/img/rouble/Rouble_1728_908_28_2.png'
rus_dict = {"Полтина": "Poltina", "Рубль": "Rouble"}

def display_2_images(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1)
    axes[0].set_title(f'Original Image\n{img1.shape}')
    axes[0].axis('off')
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'Modified Image\n{img2.shape}')
    axes[1].axis('off')
    plt.show()


def save_img_nparray(image, filename):
    image = image/255
    image_array = np.array(image)
    #flattened_image = image_array.flatten()
    filename = filename.split('.')[0] + '.npy'
    np.save(filename, image_array)


def square_img(img_path):
    image = cv.imread(img_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    circle_center_x, circle_center_y, circle_r = int(width / 2), int(height / 2), int(min(height, width) / 2 * 0.9)
    #circle_center_x, circle_center_y, circle_r = circle_detector(gray_image)
    #print(circle_center_x, circle_center_y, circle_r)
    edge_offset = 10
    square_size = min(height, width)
    #print(square_size, square_size//2)
    x1 = int(max(circle_center_x, circle_r) - min(circle_center_x, circle_r)) + edge_offset
    x2 = int(max(circle_center_x, circle_r) + min(circle_center_x, circle_r)) - edge_offset
    y1 = int(max(circle_center_y, circle_r) - min(circle_center_y, circle_r)) + edge_offset
    y2 = int(max(circle_center_y, circle_r) + min(circle_center_y, circle_r)) - edge_offset
    #print(y1, y2, x1, x2)
    cropped_image = gray_image[y1:y2, x1:x2]
    try:
        square_image = cv.resize(cropped_image, (120, 120))  # Resize to 100x100 pixels
    except:
        display_2_images(gray_image, cropped_image)
    #display_2_images(cropped_image, square_image)

    return square_image


def circle_detector(gray_image):
    def draw_circle(biggest_circle):
        center = (biggest_circle[0, 0], biggest_circle[0, 1])
        radius = biggest_circle[0, 2]
        cv.circle(gray_image, center, radius, (0, 0, 255), 3)  # Draw the outer circle
        cv.circle(gray_image, center, 2, (0, 0, 255), 3)  # Draw the center of the circle

    height, width = gray_image.shape
    circle_center_x, circle_center_y, circle_r = int(width/2), int(height/2), int(min(height, width)/2*0.9)
    height, width = gray_image.shape
    min_dimension = min(width, height)
    # Apply GaussianBlur to reduce noise and help circle detection
    blurred_image = cv.GaussianBlur(gray_image, (9, 9), 2)
    # Use HoughCircles to detect circles
    circles = cv.HoughCircles(
        blurred_image,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dimension//2,
        param1=100,
        param2=50,
        minRadius=int(min_dimension*0.15),
        maxRadius=int(min_dimension*0.495)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        max_row_index = np.argmax(circles[:, :, 2]) # Find the index of the maximum value in the third column
        biggest_circle = circles[:, max_row_index, :] # Extract the row with the maximum value
        #print(biggest_circle.shape, biggest_circle)
        circle_center_x = int(biggest_circle[0, 0])
        circle_center_y = int(biggest_circle[0, 1])
        circle_r = int(biggest_circle[0, 2])
        draw_circle(biggest_circle) # Draw detected circles

    else:
        print("No circles detected")

    return circle_center_x, circle_center_y, circle_r


def file_rename_rus_eng(rus_dict):
    for key, value in rus_dict.items():
        new_filename = filename.replace(key, value)
        old_path = os.path.join(input_folder, filename)
        new_path = os.path.join(input_folder, new_filename)
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            pass


def img_prep(input_folder, output_folder, output_folder_NPY, year_1, year_2):
    # Loop through all files in the input folder
    count = 0
    time_start = time.time()
    years_to_save = [y for y in range(year_1, year_2 + 1)]
    print(f"Years to save: {years_to_save}\n")
    for filename in os.listdir(input_folder):
        if int(filename.split('_')[1][:4]) not in years_to_save:
            #pass
            continue
        else:
            count += 1
            image_path = os.path.join(input_folder, filename)
            print(f'{count}  {filename}')
            square_image = square_img(image_path)
            cv.imwrite(output_folder + 'GS_' + filename, square_image)
            save_img_nparray(square_image, output_folder_NPY + 'GS_' + filename)
        #if count == 10: break

    delta_time = time.time() - time_start
    print(f'\nScript took {delta_time:.3f} sec')


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


def switch_sides_2(input_folder):
    for filename in os.listdir(input_folder):
        side = int(filename.split('.')[0][-1])
        file1_path = os.path.join(input_folder, filename)
        if os.path.isfile(file1_path):
            if not os.path.exists(input_folder + 'new_1/'):
                os.makedirs(input_folder + 'new_1/')
            if not os.path.exists(input_folder + 'new_2/'):
                os.makedirs(input_folder + 'new_2/')
            if side == 1:
                file2_path = os.path.join(input_folder + 'new_2/',
                                          filename.split('.')[0][:-1] + '2.' + filename.split('.')[1])
            else:
                file2_path = os.path.join(input_folder + 'new_1/',
                                          filename.split('.')[0][:-1] + '1.' + filename.split('.')[1])
            os.rename(file1_path, file2_path)


def npy_concat(input_folder_NPY, output_filename_img, output_filename_y):
    for file in [f'{output_filename_img}.npy', f'{output_filename_y}.npy']:
        print(file)
        file_path = os.path.join(input_folder_NPY, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    count = 0
    flattened_images = []
    num_files = len(os.listdir(input_folder_NPY))
    y_vector = np.zeros((num_files, 1), dtype=int)
    print(y_vector.shape)
    for filename in os.listdir(input_folder_NPY):
        loaded_array = np.load(input_folder_NPY + filename)
        print(f'{count}  {filename} {loaded_array.shape}')
        flattened_image = loaded_array.flatten()
        flattened_images.append(flattened_image)
        y_vector[count] = int(filename.split('.')[0][-1]) - 1
        count += 1

    concatenated_images = np.array(flattened_images)
    np.save(input_folder_NPY + output_filename_img, concatenated_images)
    np.save(input_folder_NPY + output_filename_y, y_vector)
    print(f'\n{concatenated_images.shape}\n'
          f'{count} images have been concatenated into {output_filename_img}.npy\n')
    print(f'{y_vector.shape}\n'
          f'{count} outputs have been saved into {output_filename_y}.npy')

#circle_center_x, circle_center_y, circle_r = circle_detector(img_path)
#print(circle_center_x, circle_center_y, circle_r)

# Ensure the output folder exists, create it if necessary
# Input and output folders
input_folder = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_Raw/'
output_folder = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_Grey_Squared/'
output_folder_NPY = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_NPY/'
switch_folder = imp_aug_dir = 'D:/_PROJECTS/Coins_Classification/Images/_TEST/aug/'

# Ensure the output folder exists, create it if necessary
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_folder_NPY):
    os.makedirs(output_folder_NPY)
if not os.path.exists(switch_folder):
    os.makedirs(switch_folder)

#square_img(img_path)
#img_prep_folder(input_folder, coin_side=2)

#for filename in os.listdir(input_folder):
#    file_rename_rus_eng(rus_dict)
switch_sides('D:/_PROJECTS/Coins_Classification/Images/_TEST/Roubles_Raw/11/')
#switch_sides_2(input_folder)
#img_prep(input_folder, output_folder, output_folder_NPY, 1700, 1799)
#npy_concat(output_folder_NPY, '_Dataset', '_y')