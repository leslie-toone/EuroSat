# needed to download a subset of Eurosat data to run on Coursera Project
#found this code at
# https://colab.research.google.com/github/e-chong/Remote-Sensing/blob/master/EuroSAT%20Land%20Cover%20Classification/EuroSAT%20Land%20Use%20and%20Land%20Cover%20Classification%20using%20Deep%20Learning.ipynb

# processing and reading images
import zipfile
import requests
import io
from PIL import Image
from numpy import asarray
from numpy import save


# tensor processing
import numpy as np
from sklearn.utils import shuffle

# plotting
import matplotlib.pyplot as plt

# modeling
from sklearn.model_selection import train_test_split
import keras

# RGB file URL
url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

# download zip
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
# get file names
txtfiles = []
for file in z.namelist():
  txtfiles.append(file)

# keep only those containing ".jpg"
txtfiles = [x for x in txtfiles if ".jpg" in x]

# read images to numpy array
XImages = np.zeros([len(txtfiles), 64, 64, 3])

i = 0
for pic in txtfiles:
  XImages[i] = np.asarray(Image.open(z.open(pic))).astype('uint8')/255
  i += 1

del r # clear memory
del z

# Get labels in numpy array as strings
labs = np.empty(len(txtfiles), dtype = 'S20')

i = 0
for label in txtfiles:
  labs[i] = label.split('/')[1]
  i += 1

# change them to integers in alphabetical order
label_names, yLabels = np.unique(labs, return_inverse=True)

label_Dict = dict(zip(np.unique(yLabels), label_names))
print(label_Dict)
np.array(np.unique(yLabels, return_counts=True)).T

# test that the labels and images read in properly
tmp = 18000
img = XImages[tmp]
print(yLabels[tmp])
print(label_names[yLabels[tmp]])
plt.imshow(img)
plt.show()
# find the smallest class
smallest_class = np.argmin(np.bincount(yLabels))
smallest_class
# number of classes
num_classes = len(np.array(np.unique(yLabels)))
# observations in smallest class
smallest_class_obs = np.where(yLabels == smallest_class)[0]

# Get 2000 observations from each class
indBal = np.empty(0, dtype=int)
for i in range(num_classes):
  indTemp = shuffle(np.where(yLabels == i)[0], random_state=42)[0:smallest_class_obs.shape[0]]
  indBal = np.concatenate([indBal, indTemp])

# shuffle the balanced index
indBal = shuffle(indBal, random_state = 42)

yBal = yLabels[indBal]
XBal = XImages[indBal]

print(yBal.shape)
print(XBal.shape)
# first line uses balanced labels
# second line uses original imbalanced labels

x_train, x_test, y_train, y_test = train_test_split(XBal, yBal, stratify = yBal, test_size = 0.2, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(XImages, yLabels, stratify = yLabels, test_size = 0.2, random_state=42)

# test that the labels and images are still matched up properly
tmp = 7000
img = x_train[tmp]
print(label_names[y_train[tmp]])
plt.imshow(img)
plt.show()

# class distribution for yTrain
print(np.array(np.unique(y_train, return_counts=True)).T)

# class distribution for yTest
print(np.array(np.unique(y_test, return_counts=True)).T)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# save to npy file
save('data/y_train.npy', y_train)
save('data/y_test.npy', y_test)
save('data/x_train.npy', x_train)
save('data/x_test.npy', x_test)