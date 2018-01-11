import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def show_hsv_images():
    for test_img in glob.glob('test_images/test*.jpg'):
        orig_img = cv2.imread(test_img)
        converted_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
        fig, axes = plt.subplots(4, 2)
        ax1d = axes.flatten()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # ax1d[0].imshow(converted_img[:, :, 1] / 2 + converted_img[:, :, 2] / 2, cmap='gray')
        ax1d[0].axis('off')
        ax1d[1].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), aspect='auto')
        ax1d[1].set_title("Original image")
        ax1d[1].axis('off')
        line = Line2D([0, 1280], [500, 500])
        ax1d[1].add_line(line)
        ax1d[2].imshow(converted_img[:,:,0], cmap='gray')
        ax1d[2].set_title("Hue channel")
        ax1d[2].axis('off')
        ax1d[2].imshow(converted_img[:,:,0], cmap='gray')
        ax1d[3].set_title("Hue channel histogram on line")
        x = np.linspace(0, 1, 1280)
        y1 = converted_img[500,:,0]
        ax1d[3].plot(x, y1)
        ax1d[4].imshow(converted_img[:,:,1], cmap='gray')
        ax1d[4].set_title("Saturation channel")
        ax1d[4].axis('off')
        ax1d[5].set_title("Saturation channel histogram on line")
        y2 = converted_img[500,:,1]
        ax1d[5].plot(x, y2)
        ax1d[6].imshow(converted_img[:,:,2], cmap='gray')
        ax1d[6].set_title("Value channel")
        ax1d[6].axis('off')
        ax1d[7].set_title("Value channel histogram on line")
        y3 = converted_img[500,:,2]
        ax1d[7].plot(x, y3)
        _ = plt.show(block=True)


def show_hls_images():
    for test_img in glob.glob('test_images/test*.jpg'):
        orig_img = cv2.imread(test_img)
        converted_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HLS)
        fig, axes = plt.subplots(4, 2)
        ax1d = axes.flatten()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        ax1d[0].axis('off')
        ax1d[1].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), aspect='auto')
        ax1d[1].set_title("Original image")
        ax1d[1].axis('off')
        line = Line2D([0, 1280], [500, 500])
        ax1d[1].add_line(line)
        ax1d[2].imshow(converted_img[:,:,0], cmap='gray')
        ax1d[2].set_title("Hue channel")
        ax1d[2].axis('off')
        ax1d[2].imshow(converted_img[:,:,0], cmap='gray')
        ax1d[3].set_title("Hue channel histogram on line")
        x = np.linspace(0, 1, 1280)
        y1 = converted_img[500,:,0]
        ax1d[3].plot(x, y1)
        ax1d[4].imshow(converted_img[:,:,1], cmap='gray')
        ax1d[4].set_title("Lightness channel")
        ax1d[4].axis('off')
        ax1d[5].set_title("Lightness channel histogram on line")
        y2 = converted_img[500,:,1]
        ax1d[5].plot(x, y2)
        ax1d[6].imshow(converted_img[:,:,2], cmap='gray')
        ax1d[6].set_title("Saturation channel")
        ax1d[6].axis('off')
        ax1d[7].set_title("Saturation channel histogram on line")
        y3 = converted_img[500,:,2]
        ax1d[7].plot(x, y3)
        _ = plt.show(block=True)


show_hsv_images()