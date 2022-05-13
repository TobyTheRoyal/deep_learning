import os

from keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.applications.densenet import layers

from dataset_creator import *

# -----------------------------------------------------------------------------
# --                       global variables
# -----------------------------------------------------------------------------

data_file_name = "distord"

models_folder = "./models/"
plots_folder = "./plots/"
cifar_example_folder = plots_folder + "cifar_examples/"
dist_example_folder = plots_folder + "distortion_examples/"

models_mini_folder = models_folder + "mini/"
models_small_folder = models_folder + "small/"
models_big_folder = models_folder + "big/"
models_huge_folder = models_folder + "huge/"

plots_mini_folder = models_mini_folder + "plots/"
plots_small_folder = models_small_folder + "plots/"
plots_big_folder = models_big_folder + "plots/"
plots_huge_folder = models_huge_folder + "plots/"

directories = [models_folder, plots_folder, cifar_example_folder, dist_example_folder,
               models_mini_folder, models_small_folder, models_big_folder, models_huge_folder,
               plots_mini_folder, plots_small_folder, plots_big_folder, plots_huge_folder]

extensions = ["gauss", "ocl", "br", "rot", "flip", "color", "position", "all"]
sizes = ["mini", "small", "big", "huge"]

epochs = 5


# -----------------------------------------------------------------------------
# --                       Models
# -----------------------------------------------------------------------------

def create_autoencoder_mini():
    input = layers.Input(shape=(32, 32, 3))  # size is 32 * 32 * 3 = 3072

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)  # 32, 32, 32
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 16, 16, 32
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)      # 16, 16, 16
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 8, 8, 16
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)      # 8, 8, 16
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 4, 4, 16

    # code size is 4 * 4 * 16 = 256

    # Decoder
    x = layers.Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same')(x)      # 4, 4, 16
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 8, 8 ,16
    x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)      # 8, 8, 32
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 16, 16 ,32
    x = layers.Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same')(x)      # 16, 16, 16
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 32, 32 ,16
    x = layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same')(x)  # 32, 32, 3

    autoencoder = Model(input, x)
    autoencoder.compile(optimizer='adamax', loss='mse')

    print(autoencoder.summary())

    return autoencoder


# -----------------------------------------------------------------------------

def create_autoencoder_small():
    input = layers.Input(shape=(32, 32, 3))  # size is 32 * 32 * 3 = 3072

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)  # 32, 32, 32
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 16, 16, 32
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)      # 16, 16, 16
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 8, 8, 16

    # code size is 8 * 8 * 16 = 1024

    # Decoder
    x = layers.Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same')(x)      # 8, 8, 16
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 16, 16 ,16
    x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)      # 16, 16, 32
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 32, 32 ,32
    x = layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same')(x)  # 32, 32, 3

    autoencoder = Model(input, x)
    autoencoder.compile(optimizer='adamax', loss='mse')

    print(autoencoder.summary())

    return autoencoder


# -----------------------------------------------------------------------------

def create_autoencoder_big():
    input = layers.Input(shape=(32, 32, 3))  # size is 32 * 32 * 3 = 3072

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)  # 32, 32, 32
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)      # 32, 32 ,64
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 16, 16, 64
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)      # 16, 16, 32
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)      # 16, 16, 16

    # code size is 16 * 16 * 16 = 4096

    # Decoder
    x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)      # 16, 16, 32
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 32, 32 ,32
    x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)      # 32, 32, 64
    x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)      # 32, 32, 32
    x = layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same')(x)  # 32, 32, 3

    autoencoder = Model(input, x)
    autoencoder.compile(optimizer='adamax', loss='mse')

    print(autoencoder.summary())

    return autoencoder


# -----------------------------------------------------------------------------

def create_autoencoder_huge():
    input = layers.Input(shape=(32, 32, 3))  # size is 32 * 32 * 3 = 3072

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)  # 32, 32, 32
    x = layers.UpSampling2D(size=(2, 2))(x)                                  # 64, 64, 32
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)      # 64, 64 ,64
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 32, 32, 64
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)      # 32, 32, 32
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)              # 16, 16, 32

    # code size is 16 * 16 * 32 = 8192

    # Decoder
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 32, 32, 32
    x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)      # 32, 32, 64
    x = layers.UpSampling2D(size=(2, 2))(x)                                                  # 64, 64 ,64
    x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)      # 64, 64, 64
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)                              # 32, 32, 64
    x = layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)      # 32, 32, 32
    x = layers.Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same')(x)  # 32, 32, 3

    autoencoder = Model(input, x)
    autoencoder.compile(optimizer='adamax', loss='mse')

    print(autoencoder.summary())

    return autoencoder


# -----------------------------------------------------------------------------
# --                       training and testing
# -----------------------------------------------------------------------------


def make_test_print_model(data, extension, size):
    name = "model_" + extension + ".h5"
    train_model(extension, size, data["x_train_" + extension], data["y_train_" + extension],
                data["x_test_" + extension],
                data["y_test_" + extension])
    model = load_model(name, size)
    test_image_print(model, data["x_test_" + extension], data["y_test_" + extension], extension, size)


# -----------------------------------------------------------------------------

def test_image_print(model, x_test, y_test, extension, size):
    random_pic_index = random.sample(range(len(x_test)), 3)
    predictions = model.predict(x_test)

    fig, ax = plt.subplots(3, 3)
    fig.suptitle(extension + " model")
    ax[0][0].set_title("Original")
    ax[0][1].set_title("Distorted")
    ax[0][2].set_title("Predicted")
    for i, r_i in enumerate(random_pic_index):
        ax[i][0].imshow(y_test[r_i])
        ax[i][1].imshow(x_test[r_i])
        ax[i][2].imshow(predictions[r_i])
    plt.savefig(get_model_plots_folder(size) + "test_images_" + extension)
    plt.clf()


# -----------------------------------------------------------------------------

def train_model(extension, size, x_train, y_train, x_test, y_test, plot=True):
    model_file = get_model_folder(size) + "model_" + extension + ".h5"
    if os.path.exists(model_file):
        print("Model already exists!")
        return

    autoencoder = get_model(size)

    checkpoint = ModelCheckpoint(model_file, verbose=1, save_best_only=True, save_weights_only=True)

    history = autoencoder.fit(x_train,
                              y_train,
                              validation_split=0.2,
                              epochs=epochs,
                              batch_size=128,
                              callbacks=[checkpoint])

    score = autoencoder.evaluate(x_test, y_test)

    if plot:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.scatter(epochs - 1, score, c="red", marker="*")
        plt.title("model_" + extension + " loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "val", "test"], loc="upper left")
        plt.grid(axis="y")
        plt.savefig(get_model_plots_folder(size) + "model_" + extension + "_loss")
        plt.clf()

    return autoencoder


# -----------------------------------------------------------------------------
# --                       loading functions
# -----------------------------------------------------------------------------

def load_model(name, size):
    dae = get_model(size)
    dae.load_weights(get_model_folder(size) + name)
    return dae


# -----------------------------------------------------------------------------

def load_data(plot=False):
    # Load the dataset
    with open(data_file_name + ".pickle", "rb") as file:
        data = pickle.load(file)

    # plot example images for each distortion
    if plot:
        for ext in extensions:
            # plot first few images
            for i in range(3):
                # define subplot
                plt.subplot(2, 3, 1 + i)
                # plot raw pixel data
                plt.imshow(data['y_train_' + ext][i])
            # plot first few images
            for i in range(3):
                # define subplot
                plt.subplot(2, 3, 4 + i)
                # plot raw pixel data
                plt.imshow(data['x_train_' + ext][i])
            plt.savefig(dist_example_folder + "example_" + ext)
            plt.clf()
    return data


# -----------------------------------------------------------------------------
# --                       size helper functions
# -----------------------------------------------------------------------------

def get_model_folder(size):
    if size == "mini":
        return models_mini_folder
    elif size == "small":
        return models_small_folder
    elif size == "big":
        return models_big_folder
    else:
        return models_huge_folder


# -----------------------------------------------------------------------------

def get_model_plots_folder(size):
    if size == "mini":
        return plots_mini_folder
    elif size == "small":
        return plots_small_folder
    elif size == "big":
        return plots_big_folder
    else:
        return plots_huge_folder


# -----------------------------------------------------------------------------

def get_model(size):
    if size == "mini":
        return create_autoencoder_mini()
    elif size == "small":
        return create_autoencoder_small()
    elif size == "big":
        return create_autoencoder_big()
    else:
        return create_autoencoder_huge()


# -----------------------------------------------------------------------------
# --                       helper functions
# -----------------------------------------------------------------------------

def create_directories():
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)


# -----------------------------------------------------------------------------

def create_distorted_dataset(data_creator, plot=False):
    if plot:
        # plot first few images
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(data_creator.x_train[i])
        # show the figure
        plt.savefig(cifar_example_folder + "cifar_examples")
        plt.clf()

    data_creator.create_data(data_file_name)


# -----------------------------------------------------------------------------
# --                       main function
# -----------------------------------------------------------------------------

def main():
    create_directories()
    # Create the dataset with distorted images
    data_creator = DataCreator()
    if not os.path.exists("./" + data_file_name + ".pickle"):
        create_distorted_dataset(data_creator, plot=True)

    # Load the dataset
    data = load_data(plot=True)

    # create, train and test the models
    for size in sizes:
        for ext in extensions:
            make_test_print_model(data, ext, size)

    print("Done!")


if __name__ == "__main__":
    main()
