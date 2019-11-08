import cv2
import glob
import numpy as np
# from utils import salva_imagem_com_predicao
import pandas as pd
import matplotlib.pyplot as plt
from skimage import transform
from PIL import Image
import os

X = []
y = []


def get_dataset(path_image, Cat=True):
    images = glob.glob(path_image+"*.png")
    for image in images:
        img = Image.open(image)
        img = np.asarray(img)
        if Cat:
            y.append(1)
        else:
            y.append(0)
        img = transform.resize(img, (68, 68))
        img = img.flatten()
        X.append(img)
    

arquivos_de_gatos = "train/cat/"
arquivos_nao_gatos = "train/noncat/"

get_dataset(arquivos_de_gatos)
get_dataset(arquivos_nao_gatos, False)

X = np.asarray(X)
Y = np.asarray(y)
print(X.shape)
print(Y.shape)
m, n = X.shape
X = X/255
print(m, n)

X = np.hstack((np.ones((m, 1)), X))
print(X.shape)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(X, theta):
    return sigmoid(X.dot(theta))

def cost_function(theta, X, y):
    m = y.shape[0]
    y_zero = (1 - y).dot(np.log(1 - h(X, theta)))
    y_one = y.dot(np.log(h(X, theta)))
    J = (-1 / m) * (y_zero + y_one)
    return J


def gradient(theta, X, y):
    m = len(y)
    return (h(X, theta) - y).dot(X) / m


def predict(X, theta):
    return np.round(h(X, theta))


initial_theta = np.ones(n+1)

# Compute and display initial cost and gradient.
cost = cost_function(initial_theta, X, Y)
grad = gradient(initial_theta, X, y)
print("Cost at initial theta (zeros): {}".format(cost))
# print("Expected cost (approx): 0.693")
print("Gradient at initial theta (zeros):")
print(grad)
it = 0
for i in X:
    prob = sigmoid(i).dot(initial_theta)
    print("Predicted: {}, Truth: {}".format(prob, y[it]))
    it += 1
# Calculate accuracy of the algorithm on the training set.
p = predict(X, initial_theta)

print('Train Accuracy: {}'.format(np.mean((p == y)) * 100))
