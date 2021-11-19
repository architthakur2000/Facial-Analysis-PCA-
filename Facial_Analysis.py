from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):

    #This function loads the file.

    x = np.load(filename)
    x = x - np.mean(x, axis = 0)
    return x

def get_covariance(dataset):

    #This function performs the mathematical calculation of covariance.

    multiply = np.dot(np.transpose(dataset), dataset)
    req_length_ds = len(dataset) - 1
    covariance = multiply / req_length_ds
    return covariance

def get_eig(S, m):

    #This function get's the eigen values.

    x, y = eigh(S, eigvals = (len(S) - m, len(S) - 1))
    z = np.flip(np.argsort(x))
    return np.diag(x[z]), y[:, z]

def get_eig_perc(S, perc):

    #This function get's the eigen value percentage

    x,y = eigh(S)
    percentage = np.sum(x) * perc
    x_1, y_1 = eigh(S, eigvals = (percentage, np.inf))
    j = np.flip(np.argsort(x_1))
    return np.diag(x_1[j]), y_1[:, j]

def project_image(img, U):

    #this function projects the image.

    mult_1 = np.dot(np.transpose(U), img)
    ret = np.dot(U, mult_1)
    return ret

def display_image(orig, proj):

    # This function displays the image.
    orig = np.reshape(orig, (32,32), order = 'F')
    proj = np.reshape(proj, (32,32), order = 'F')

    figure, (dx1, dx2) = plt.subplots(nrows = 1, ncols = 2)
    dx1.set_title('Original')
    dx2.set_title('Projection')

    map_dx1 = dx1.imshow(orig, aspect = 'equal')
    figure.colorbar(map_dx1, ax = dx1)
    map_dx2 = dx2.imshow(proj, aspect = 'equal')
    figure.colorbar(map_dx2, ax = dx2)
    plt.savefig("test.png")
    plt.show()
