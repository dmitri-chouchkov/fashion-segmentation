import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.cm as colormaps

from PIL import Image as im
import numpy as np
from scipy.signal import convolve2d



def expand(x: np.ndarray, itterations = 1) ->np.ndarray:
    kernel = np.ones([ 2* itterations + 1, 2 * itterations + 1],dtype=np.uint8)
    return (convolve2d(x.astype(np.uint8),kernel,mode='full') >= 1).astype(np.uint8)

def contract(x: np.ndarray,itterations = 1) -> np.ndarray:
    size = 2 * itterations + 1
    kernel = np.ones([ size, size],dtype=np.uint8)
    for i in range(itterations):
        for j in range(itterations):
            if i < itterations - j:
                kernel[i, j] = 0
                kernel[size -i - 1, j] = 0
                kernel[i, size -j -1] = 0
                kernel[size -i - 1, size -j -1] = 0
    print(kernel)
    return (convolve2d(1 - x.astype(np.uint8),kernel,mode='valid') == 0).astype(np.uint8)

def hollow(x: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,0,1,1],[0,1,1,1,0],[0,0,1,0,0]], dtype=np.uint8) 
    interior = convolve2d(x, kernel,mode='same',boundary='fill', fillvalue=1)
    output = x.copy()
    output[interior == 12] = 0
    return output

def main():

    # setup test
    test = np.zeros([128,128], dtype= np.uint8) 
    test[0,:] = 1
    test[1,:] = 1
    test[-1, :] = 1
    test[-2, :] = 1
    test[:, 0] = 1
    test[:, 1] = 1
    test[:, -1] = 1
    test[:, -2] = 1

    x_1 = 20
    y_1 = 20

    x_2 = 22
    y_2 = 31
    test[y_1:y_1 + 2,x_1:x_1 + 2] = 1

    test[y_2:y_2 + 2,x_2:x_2 + 2] = 1

    src = test

    src_exp = expand(src, 5)

    src_exp_ctrct = contract(src_exp, 5)

    src_exp_ctrct_hollow = hollow(src_exp_ctrct)

    ax = []


    # plot results

    fig = plt.figure(figsize=(512.0 * 4/196, 512.0/196), dpi=196)
    rows = 1
    columns = 4

    ax.append( fig.add_subplot(rows, columns, 1))
    ax[-1].set_title('src')
    plt.imshow(im.fromarray(src))
    
    ax.append( fig.add_subplot(rows, columns, 2))
    ax[-1].set_title('expand')
    plt.imshow(im.fromarray(src_exp, 'L'), interpolation='none')
    
    ax.append( fig.add_subplot(rows, columns, 3))
    ax[-1].set_title('contract')
    plt.imshow(im.fromarray(src_exp_ctrct,'L'), interpolation='none')

    ax.append( fig.add_subplot(rows, columns, 4))
    ax[-1].set_title('hollow')
    plt.imshow(im.fromarray(src_exp_ctrct_hollow,'L'), interpolation='none')

    plt.show()




if __name__ == "__main__":
    main()



