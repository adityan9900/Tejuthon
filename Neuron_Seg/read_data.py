import os
import numpy as np
from numpy import load
from PIL import Image
import cv2
from extract_patches import ExtractPatches_test



# Load fruit fly data for training.
data = load('./data/fruit_fly_volumes.npz')
train_volume = np.expand_dims(data['volume'], axis=-1)
train_label = np.expand_dims(data['label'], axis=-1)


num_images = train_volume.shape[0]
total_count = 0
for i in range (num_images):
    current_im = np.squeeze(train_volume[i, :, :, :], axis=2)
    current_im = np.stack((current_im)*3, axis=-1)

    current_label = np.expand_dims(train_label[i, :, :, :], axis=2)



    im_iter = ExtractPatches_test((128,128), (300, 300), current_im)
    label_iter = ExtractPatches_test((128,128), (300, 300), current_label)



    j = 0
    for im_patch, label_patch in zip(im_iter, label_iter):

        im_patch = np.squeeze(im_patch)
        label_patch = np.squeeze(label_patch)


        print("yeeee: ", im_patch.shape, label_patch.shape)
        im_patch = Image.fromarray(im_patch, 'L')
        label_patch = Image.fromarray(label_patch, 'L')

        if(total_count < 500):
            im_patch.save(os.path.join("./data/fruit_fly/image/",  str(i) + "_" + str(j) + ".png"))
            label_patch.save(os.path.join("./data/fruit_fly/label/", str(i) + "_" + str(j) + ".png"))

        elif(total_count < 750):
            im_patch.save(os.path.join("./data/fruit_fly/image_val/",  str(i) + "_" + str(j) + ".png"))
            label_patch.save(os.path.join("./data/fruit_fly/label_val/", str(i) + "_" + str(j) + ".png"))

        total_count += 1
        j += 1
#ExtractPatches_test(sizeInputPatch, stride, image)



"""




print("fruit fly volumes: ", train_label.shape)

# Load mouse data for evaluation.
data = load('./data/mouse_volumes.npz')
test_volume = np.expand_dims(data['volume'], axis=-1)
test_label = np.expand_dims(data['label'], axis=-1)


from scipy import misc
img = np.squeeze(test_label[0,:,:,:], axis=2)
img[img==1] = 255
#misc.imshow(img)


#print("mouse volumes: ", test_volume.shape)
#print("yuh: ", train_volume[0,:5,:5,:])

#cv2.imshow('yeh', train_volume[0,:,:,:])
img = Image.fromarray(img, 'L')
img.show()
"""
