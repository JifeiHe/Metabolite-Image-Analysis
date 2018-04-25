import numpy as np
import random, os, time, cv2
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.python.client import device_lib
from datetime import datetime
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
# import ipywidgets as ipywidgets
# from ipywidgets import interact
from matplotlib import cm
from PIL import Image
import keras
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
todate = datetime.now().strftime('%Y%m%d')

def load_data(i_img):
    data_dir = '/'
    os.chdir(data_dir)
    img1 = np.loadtxt('./Results-NP_C3_N/Images.txt')[i_img]
    return img1

def data_preprocess(img):
    return img if max(img)==0 else img/max(img)*1.0

def white_noise(img, img_col, img_row, intensity):
    img_tmp=[]
    for i in range(len(img)):
        img_tmp.append(img[i]+intensity*0.01*img[i]*np.random.normal(0, 1))    
    img_tmp = np.asarray(img_tmp)
    return img_tmp

def image_transform(img, img_col, img_row):
    img_resize = cv2.resize(img.reshape((img_col, img_row)), (224, 224), interpolation = cv2.INTER_LINEAR)
    img_add_channel = Image.fromarray(np.uint8(cm.jet(img_resize)*255)).convert("RGB")
    return np.array(img_add_channel)

def choose_model(which_model):
    image_input = Input(shape=(224, 224, 3))
    if which_model == 'vgg16':
        model = VGG16(input_tensor=image_input, weights='imagenet', include_top=True)
    elif which_model == 'vgg19':
        model = VGG19(input_tensor=image_input, weights='imagenet', include_top=True)
    elif which_model == 'inception_v3':
        model = InceptionV3(input_tensor=image_input, weights='imagenet', include_top=True)
    elif which_model == 'resnet50':
        model = ResNet50(input_tensor=image_input, weights='imagenet', include_top=True)
    base_pred = model.get_layer('fc2').output
    base_model = Model(image_input, base_pred)
    # print(base_model.summary())
    return base_model, which_model

def generate4096(img1_trans, img1_sft_trans):
    base_model, which_model = choose_model(which_model='vgg16')
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    base_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # print('start timing...')
    start_timing = time.time()
    img1_4096 = base_model.predict(img1_trans)
    img1_sft_4096 = base_model.predict(img1_sft_trans)
    print('training time: {0:.4f} seconds'.format(time.time() - start_timing))
    return img1_4096, img1_sft_4096

# ----------------------------------------------------------------------------

def main_vgg(i_img=27, intensity=20, save=False):
    img1 = load_data(i_img)
    img1 = data_preprocess(img1)
    img_col, img_row = 60, 12
    # shift left pixels
    img1_sft = white_noise(img1, img_col, img_row, intensity)
    print('shift {}px left in img1'.format(intensity))
    # plot image
    plt.subplot(121)
    plt.imshow(img1.reshape((img_col, img_row)), cmap='jet')
    plt.title('img1')
    plt.subplot(122)
    plt.imshow(img1_sft.reshape((img_col, img_row)), cmap='jet')
    plt.title('img1_white_noise')
    # plt.show()
    # matrix norm
    img1_mn = img1 / np.linalg.norm(img1)
    img1_sft_mn = img1_sft / np.linalg.norm(img1_sft)
    simi_720 = np.dot(img1_mn, img1_sft_mn)
    # plot images after resize to fit vgg16
    img1_trans = image_transform(img1, img_col, img_row)
    img1_sft_trans = image_transform(img1_sft, img_col, img_row)
    vgg_img_col, vgg_img_row = 224, 224
    # plt.subplot(121)
    # plt.imshow(img1_trans, cmap='jet')
    # plt.title('img1')
    # plt.subplot(122)
    # plt.imshow(img1_sft_trans, cmap='jet')
    # plt.title('img1_sft')
    # plt.show()
    # add one dimension
    img1_trans = np.expand_dims(img1_trans, axis=0)
    img1_sft_trans = np.expand_dims(img1_sft_trans, axis=0)
    # print(img1_trans.shape)
    # print(img1_sft_trans.shape)
    # generate vgg-ready image
    img1_4096, img1_sft_4096 = generate4096(img1_trans, img1_sft_trans)
    img1_4096 = np.squeeze(img1_4096, axis=0)
    img1_sft_4096 = np.squeeze(img1_sft_4096, axis=0)
    # matrix norm
    img1_4096_mn = img1_4096 / np.linalg.norm(img1_4096)
    img1_sft_4096_mn = img1_sft_4096 / np.linalg.norm(img1_sft_4096)
    simi_4096 = np.dot(img1_4096_mn, img1_sft_4096_mn)
    # print('similarity level w/out vgg: {0:.4f}'.format(simi_720))
    # print('similarity level w/ vgg: {0:.4f}'.format(simi_4096))
    return simi_720, simi_4096

def simi_loop(i_img=27, intensity=20, save=False):
    simi_720_lst, simi_4096_lst = [], []
    for i in range(intensity):
        simi_720, simi_4096 = main_vgg(i_img, intensity=i)
        simi_720_lst.append(simi_720)
        simi_4096_lst.append(simi_4096)
    # plot similarity scatter plots
    plt.figure(figsize=(10,4))
    plt.scatter(range(intensity), simi_720_lst, c='blue', label='w/out vgg')
    plt.scatter(range(intensity), simi_4096_lst, c='red', label='w/ vgg')
    plt.xticks(np.arange(0,intensity,1))
    plt.yticks(np.arange(0, 1.2, .2))
    plt.xlabel('# of white noise%')
    plt.ylabel('similarity level')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()
    return simi_720_lst, simi_4096_lst

# i_img = image index; num_sft = maximum shifts allowed
simi_720_lst_27, simi_4096_lst_27 = simi_loop(i_img=4, intensity=20)
