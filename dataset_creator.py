import tensorflow as tf
import numpy as np
import pickle
from scipy.ndimage.interpolation import rotate
from matplotlib import pyplot as plt
import random

class DataCreator():

    def __init__(self):
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.cifar10.load_data()

    def create_data(self, name, noise_std=0, occlusion_size=(20, 20), brightness_range=(1.1, 1.8), angle_range=(10, 350), flip_mode='both'):
        # split the original dataset in 5 parts
        train_split = np.split(self.x_train, 5)
        test_split = np.split(self.x_test, 5)

        print("Creating training set...")

        x_gauss_train, y_gauss_train = self.additive_gaussian_noise(train_split[0], noise_std)
        x_ocl_train, y_ocl_train = self.occlusion(train_split[1], occlusion_size)
        x_br_train, y_br_train = self.brightness_change(train_split[2], brightness_range)
        x_rot_train, y_rot_train = self.rotation(train_split[3], angle_range)
        x_flip_train, y_flip_train = self.flip(train_split[4], flip_mode)

        print("Done.")

        print("Creating test set...")

        x_gauss_test, y_gauss_test = self.additive_gaussian_noise(test_split[0], noise_std)
        x_ocl_test, y_ocl_test = self.occlusion(test_split[1], occlusion_size)
        x_br_test, y_br_test = self.brightness_change(test_split[2], brightness_range)
        x_rot_test, y_rot_test = self.rotation(test_split[3], angle_range)
        x_flip_test, y_flip_test = self.flip(test_split[4], flip_mode)

        print("Done.")

        print("Exporting created dataset... ", end="")

        # save the data to a dataset
        dataset = {
            'x_train_gauss': x_gauss_train,
            'y_train_gauss': y_gauss_train,
            'x_train_ocl': x_ocl_train,
            'y_train_ocl': y_ocl_train,
            'x_train_br': x_br_train,
            'y_train_br': y_br_train,

            'x_train_rot': x_rot_train,
            'y_train_rot': y_rot_train,
            'x_train_flip': x_flip_train,
            'y_train_flip': y_flip_train,

            'x_test_gauss': x_gauss_test,
            'y_test_gauss': y_gauss_test,
            'x_test_ocl': x_ocl_test,
            'y_test_ocl': y_ocl_test,
            'x_test_br': x_br_test,
            'y_test_br': y_br_test,

            'x_test_rot': x_rot_test,
            'y_test_rot': y_rot_test,
            'x_test_flip': x_flip_test,
            'y_test_flip': y_flip_test,

            'x_train_color': x_gauss_train + x_ocl_train + x_br_train,
            'y_train_color': y_gauss_train + y_ocl_train + y_br_train,
            'x_test_color': x_gauss_test + x_ocl_test + x_br_test,
            'y_test_color': y_gauss_test + y_ocl_test + y_br_test,

            'x_train_position': x_rot_train + x_flip_train,
            'y_train_position': y_rot_train + y_flip_train,
            'x_test_position': x_rot_test + x_flip_test,
            'y_test_position': y_rot_test + y_flip_test,

            'x_train_all': x_rot_train + x_flip_train + x_gauss_train + x_ocl_train + x_br_train,
            'y_train_all': y_rot_train + y_flip_train + y_gauss_train + y_ocl_train + y_br_train,
            'x_test_all': x_rot_test + x_flip_test + x_gauss_test + x_ocl_test + x_br_test,
            'y_test_all': y_rot_test + y_flip_test + y_gauss_test + y_ocl_test + y_br_test,
        }

        # normalize the data
        for key in dataset.keys():
            dataset[key] = np.array(dataset[key]).astype('float') / 255

        with open(name + '.pickle', 'wb') as file:
            pickle.dump(dataset, file)

        print("Done")

    def apply_gaussian_noise(self, X, sigma=20):
        noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
        return X + noise

    def additive_gaussian_noise(self, data, noise_std):
        x_result = []
        y_result = []

        max_count = len(data)
        count = 0
        for image in data:
            print(f"Distorting images with aditive Gaussian noise progress: {(count / max_count) * 100.0}%", end="\r")

            if noise_std == 0:
                noise_std = np.std(image) / 2

            noise = np.random.normal(0, noise_std, image.shape)

            distorted_image = self.apply_gaussian_noise(image)

            distorted_image = np.clip(distorted_image, 0, 255)

            x_result.append(distorted_image)
            y_result.append(image)

            count += 1

        print(f"Distorting images with aditive Gaussian noise progress: {(count / max_count) * 100.0}%")

        return x_result, y_result

    def occlusion(self, data, size):
        x_result = []
        y_result = []

        max_count = len(data)
        count = 0
        for image in data:
            print(f"Distorting images by occlusion with black patches progress: {(count / max_count) * 100.0}%", end="\r")

            x = random.randint(0, image.shape[0] - 1)
            y = random.randint(0, image.shape[1] - 1)

            width = random.randint(1, size[0])
            height = random.randint(1, size[1])

            direction = random.randint(0, 3)

            x_step = 1 if direction == 1 or direction == 2 else -1
            y_step = 1 if direction == 2 or direction == 3 else -1

            distorted_image = np.copy(image)

            old_x = x
            old_y = y

            while abs(old_y - y) <= height:
                if y >= image.shape[1] or y < 0:
                    break

                while abs(old_x - x) <= width:
                    if x >= image.shape[0] or x < 0:
                        break

                    for i in range(image.shape[2]):
                        distorted_image[x][y][i] = 0
                    x += x_step
                x = old_x
                y += y_step

            x_result.append(distorted_image)
            y_result.append(image)

            count += 1

        print(f"Distorting images by occlusion with black patches progress: {(count / max_count) * 100.0}%")

        return x_result, y_result

    def brightness_change(self, data, brightness_range):
        x_result = []
        y_result = []

        max_count = len(data)
        count = 0
        for image in data:
            print(f"Distorting images by changing brightness progress: {(count / max_count) * 100.0}%", end="\r")

            if random.randint(0, 1) == 0:
                distorted_image = image.astype('float64') / random.uniform(brightness_range[0], brightness_range[1])
            else:
                distorted_image = image.astype('float64') * random.uniform(brightness_range[0], brightness_range[1])

            distorted_image = np.clip(distorted_image, 0, 255)

            distorted_image = distorted_image.astype('uint8')

            x_result.append(distorted_image)
            y_result.append(image)
            count += 1

        print(f"Distorting images by changing brightness progress: {(count / max_count) * 100.0}%")

        return x_result, y_result

    def rotation(self, data, angle_range):
        x_result = []
        y_result = []

        max_count = len(data)
        count = 0
        for image in data:
            print(f"Distorting images with rotation progress: {(count / max_count) * 100.0}%", end="\r")

            distorted_image = rotate(image, angle=random.randint(angle_range[0], angle_range[1]), reshape=False, mode='reflect')

            x_result.append(distorted_image)
            y_result.append(image)
            count += 1

        print(f"Distorting images with rotation progress: {(count / max_count) * 100.0}%")

        return x_result, y_result

    def flip(self, data, mode):
        x_result = []
        y_result = []

        max_count = len(data)
        count = 0
        for image in data:
            print(f"Distorting images by flipping progress: {(count / max_count) * 100.0}%", end="\r")

            if mode == 'horizontal':
                distorted_image = np.flip(image, axis=1)
            elif mode == 'vertical':
                distorted_image = np.flip(image, axis=0)
            else:
                distorted_image = np.flip(image, axis=0) if random.randint(0, 1) == 0 else np.flip(image, axis=1)

            x_result.append(distorted_image)
            y_result.append(image)
            count += 1

        print(f"Distorting images by flipping progress: {(count / max_count) * 100.0}%")

        return x_result, y_result
