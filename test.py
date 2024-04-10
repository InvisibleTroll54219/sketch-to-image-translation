from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
import natsort
import cv2
import os

def load_filename(path):
    dirFiles = os.listdir(path)
    for i, file in enumerate(dirFiles):
        dirFiles[i] = path + file
    return natsort.natsorted(dirFiles ,reverse=False)

# load all images in a directory into memory
def load_images(list_path, size=(256, 256)):
    img_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in list_path:
        # load and resize the image
        pixels = load_img(filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        pixels = (pixels - 127.5) / 127.5
        img_list.append(pixels)
    return np.asarray(img_list)

def pred_images(g_model, target_dir, filenames, batch_size=128):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    imgs = load_images(filenames)
    g_img = g_model.predict(imgs)
    g_img = g_img * 127.5 + 127.5
    for j, _img in enumerate(g_img):
        cv2.imwrite(target_dir + "/" + os.path.basename(filenames[j]), cv2.resize(cv2.cvtColor(_img.astype('uint8'), cv2.COLOR_RGB2BGR), (200, 250)))
    print("Image has been successfully saved in \"" + target_dir + "\" folder")
    
filenames = load_filename('Dataset/CUHK/Testing sketch/')

g_model = load_model('Models/Pixel[1]_Context[0]/g_model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})

pred_images(g_model, "Generated Images/Generated_Pixel[1]_Context[0]", filenames)

g_model = load_model('Models/Pixel[08]_Context[02]/g_model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})

pred_images(g_model, "Generated Images/Generated_Pixel[08]_Context[02]", filenames)

g_model = load_model('Models/Pixel[05]_Context[05]/g_model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})

pred_images(g_model, "Generated Images/Generated_Pixel[05]_Context[05]", filenames)

g_model = load_model('Models/Pixel[02]_Context[08]/g_model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})

pred_images(g_model, "Generated Images/Generated_Pixel[02]_Context[08]", filenames)