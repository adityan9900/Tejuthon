import os
import torch
import numpy as np
from numpy import load
from PIL import Image
from models import ResNetUNet
from extract_patches import ExtractPatches_test, MergePatches_test
from torchvision import transforms
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

model_path = os.path.join('./unet_trained_model.pkl')
unet = ResNetUNet(2)
unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

test_data  = load(os.path.join('./data/mouse_volumes.npz'))

test_images = np.expand_dims(test_data['volume'], axis=-1)
test_labels = np.expand_dims(test_data['label'], axis=-1)


total_preds = np.zeros([0])
total_labels = np.zeros([0])

print(test_images.shape)
for i in range (test_images.shape[0]):

    print("Current img:  ", i)
    current_im = test_images[i, :, :, :]
    current_label = test_labels[i, :, :, :]

    prediction_volume = np.zeros([128,128,1])

    patch_iter = ExtractPatches_test((128, 128), (128, 128), current_im)
    label_iter = ExtractPatches_test((128,128), (128,128), current_label)

    for (patch, label) in zip(patch_iter, label_iter):
        stacked_patch = np.squeeze(np.stack((patch,)*3, axis=-1))
        tensor_transform = transforms.ToTensor()
        patch_input = tensor_transform(Image.fromarray(stacked_patch))

        patch_input = patch_input.unsqueeze(0)
        result = unet(patch_input)

        result = result.detach().numpy()
        result = np.squeeze(result)


        result = np.argmax(result,axis=0)

        total_preds = np.append(total_preds, result.flatten())
        total_labels = np.append(total_labels, label.flatten())

        #result = np.expand_dims(result, axis=2)

        #if(np.sum(prediction_volume) == 0): prediction_volume = result
        #else: prediction_volume = np.concatenate((prediction_volume, result), axis=2)



    #current_pred = MergePatches_test(prediction_volume, (100,100), (384, 384), (128,128), (128,128))
print(total_preds.shape)
print(total_labels.shape)


print("AUC: ", roc_auc_score(total_labels, total_preds))
print("F1: ", f1_score(total_labels, total_preds))
print("Accuracy: ", accuracy_score(total_labels, total_preds))
