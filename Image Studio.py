import os
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from tkinter import *
from tkinter import ttk
from scipy import signal
from skimage.color import rgb2gray
from skimage.io import imread
from tkinter import filedialog
import threading

gray_image = None
filtered_image = None
original_image = None
hsv_img = None


def sel():
    global original_image
    grey = rgb2gray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    Original_plot.imshow(grey, cmap=plt.get_cmap('gray'))
    Original_Figure.canvas.draw_idle()
    return


def sel2():
    global original_image
    org = original_image
    Original_plot.imshow(org)
    Original_Figure.canvas.draw_idle()
    return


# function to reduce redundancy in code lines , creates four canvases to display images
def fig_creation(name, graph_name, wheretoput):
    name, (graph_name) = plt.subplots(1)
    canvas = FigureCanvasTkAgg(name,
                               master=wheretoput)

    # placing the canvas on the Tkinter window

    canvas.get_tk_widget().pack(fill='both', expand=True)
    return name, graph_name


def openImage():
    global original_image
    global gray_image, hsv_img
    # takes filepath and determine filetypes here are png and jpg
    filepath = filedialog.askopenfilename(
        filetypes=(('Png Files', "*.png"), ('Jpg Files', "*.jpg"), ("all files", ".*")))
    # read and display image and convert from BGR to RGB
    original_image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    Original_plot.imshow(original_image)
    # line to update canvas each time button pressed
    Original_Figure.canvas.draw_idle()

    # Convert to HSV
    hsv_img = cv2.cvtColor(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV, )

    # convert into grayscale and display it in filterd slot
    gray_image = rgb2gray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    main_thread = threading.Thread(target=hist_init)
    main_thread.start()
    Filterd_plot.imshow(gray_image, cmap=plt.get_cmap('gray'))
    Filtered_Figure.canvas.draw_idle()
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    Freq_plot.imshow(magnitude_spectrum, cmap=plt.get_cmap('gray'))
    Freq_Figure.canvas.draw_idle()

    return


def hist_init():
    Histogram_plot.hist(gray_image, bins=5)
    Histogram_Figure.canvas.draw_idle()


def filter_it_freq(img, mask):
    global filtered_image
    img_fft = np.fft.fft2(img)

    # Shift zero to the center and taking the log scale of the Fourier transform
    img_fft_shifted = np.fft.fftshift(img_fft)
    magnitude_spectrum = 20 * np.log(np.abs(img_fft_shifted))
    # Applying the filter
    img_fft_shifted_filtered = img_fft_shifted * mask
    magnitude_spectrum_filtered = magnitude_spectrum * mask

    # Inverse fshift and fft
    img_ishift = np.fft.ifftshift(img_fft_shifted_filtered)
    magnitude_spectrum_ishift = np.fft.ifftshift(magnitude_spectrum_filtered)
    filtered_image = np.abs(np.fft.ifft2(img_ishift))
    magnitude_spectrum_after_lpf = np.fft.ifft2(magnitude_spectrum_ishift)


def filter_it_spatial(filter_type, passed_pic):
    global filtered_image

    if filter_type == 1:
        highpass_kernel = np.array([[-1, -1, -1, -1, -1],
                                    [-1, 1, 2, 1, -1],
                                    [-1, 2, 4, 2, -1],
                                    [-1, 1, 2, 1, -1],
                                    [-1, -1, -1, -1, -1]])
        filtered_image = cv2.filter2D(passed_pic, -1, highpass_kernel)
    else:
        rows, cols = 5, 5
        avg_factor = 25
        lowpass_kernel = 1 / avg_factor * np.ones((rows, cols), np.uint8)
        filtered_image = cv2.filter2D(passed_pic, -1, lowpass_kernel)


def equalize(im_gray):
    intensities = np.zeros(256)  # array holds the count of intensities
    CDF = np.zeros(256)
    rows, cols = im_gray.shape
    equalized = np.zeros((rows, cols))  # will hold the final output

    for row in range(0, rows):
        for col in range(0, cols):
            val = im_gray[row][col]
            intensities[val] += 1

    PDF = intensities / (rows * cols)
    CDF[0] = PDF[0]
    for i in range(1, len(PDF)):
        CDF[i] = CDF[i - 1] + PDF[i]

    pre_equalized = (CDF * np.amax(im_gray)).astype(int)

    for row in range(0, rows):
        for col in range(0, cols):
            equalized[row][col] = pre_equalized[im_gray[row][col]]

    return equalized


# template for applying filters
def ApplyFilter(self):
    Filters_CB['values'] = ['Low pass freq domain', 'High pass freq domain', 'Low pass spatial domain',
                            'High pass spatial domain', 'Median', 'Laplacian', 'H Equalization']
    global gray_image
    global filtered_image
    global original_image, hsv_img
    filtered_image = []
    val = Filters_CB.get()
    rows, cols = gray_image.shape

    # Getting the center row and center column
    crow = int(rows / 2)
    ccol = int(cols / 2)

    # Kernel for LPF
    mask_lpf = np.zeros((rows, cols), np.uint8)
    mask_lpf[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # Kernel for HPF
    mask_hpf = np.ones((rows, cols), np.uint8)
    mask_hpf[crow - 30:crow + 30, ccol - 10:ccol + 10] = 0
    print(gray_scale.get())
    if gray_scale.get():
        if val == 'Low pass freq domain':
            filter_it_freq(gray_image, mask_lpf)

        elif val == 'High pass freq domain':
            filter_it_freq(gray_image, mask_hpf)

        elif val == 'Low pass spatial domain':
            filter_it_spatial(0, gray_image)

        elif val == 'High pass spatial domain':
            filter_it_spatial(1, gray_image)

        elif val == 'Median':
            temp_image = cv2.medianBlur(original_image, 5)
            filtered_image = rgb2gray(cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
        elif val == 'Laplacian':
            # Apply GaussianBlur To  the Gray Scale Image
            imgGus = cv2.GaussianBlur(gray_image, (3, 3), cv2.BORDER_REPLICATE)

            # Apply Laplacian To the Guass Gray Scale Image
            filtered_image = cv2.Laplacian(imgGus, cv2.CV_64F, 3, cv2.BORDER_REPLICATE)

        elif val == 'H Equalization':
            im_gray = (gray_image * 255).astype(int)  # converting it to gray scale with range 0-255
            filtered_image = equalize(im_gray)

        Original_plot.imshow(gray_image, cmap=plt.get_cmap('gray'))

    else:
        if val == 'Low pass freq domain':
            filter_it_freq(gray_image, mask_lpf)

        elif val == 'High pass freq domain':
            filter_it_freq(gray_image, mask_hpf)

        elif val == 'Low pass spatial domain':
            filter_it_spatial(0, original_image)

        elif val == 'High pass spatial domain':
            filter_it_spatial(1, original_image)

        elif val == 'Median':
            filtered_image = cv2.medianBlur(original_image, 5)
            # filtered_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)

        elif val == 'Laplacian':
            # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #RGB Image
            hsv_img_hadeer = hsv_img.copy()
            hsv_img_hadeer[:, :, 2] = cv2.Laplacian(hsv_img[:, :, 2], cv2.CV_8U, 3, cv2.BORDER_REPLICATE)
            HSV2GBR_img = cv2.cvtColor(hsv_img_hadeer, cv2.COLOR_HSV2BGR)
            filtered_image = cv2.cvtColor(HSV2GBR_img, cv2.COLOR_BGR2RGB)  # RGB Image
            print(id(hsv_img_hadeer), id(hsv_img))

        elif val == 'H Equalization':
            new_v = equalize(hsv_img[:, :, 2])
            hsv_img_noran = hsv_img.copy()
            hsv_img_noran[:, :, 2] = new_v
            filtered_image = cv2.cvtColor(hsv_img_noran, cv2.COLOR_HSV2BGR)
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
            print(id(hsv_img_noran), id(hsv_img))

        Original_plot.imshow(original_image, cmap=plt.get_cmap('gray'))
    thread_1 = threading.Thread(target=drawing_the_frequency_spectra)
    thread_2 = threading.Thread(target=draw_filtered_image)
    thread_3 = threading.Thread(target=draw_histogram)
    thread_3.start()
    thread_2.start()
    thread_1.start()
    return


def draw_histogram():
    global filtered_image
    if len(filtered_image.shape) == 3:
        Histogram_plot.clear()
        Histogram_plot.hist(cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY), bins=5)
    else:
        Histogram_plot.clear()
        Histogram_plot.hist(filtered_image, bins=5)
    Histogram_Figure.canvas.draw_idle()


def draw_filtered_image():
    global filtered_image
    Filterd_plot.imshow(filtered_image, cmap=plt.get_cmap('gray'))
    Filtered_Figure.canvas.draw_idle()


def drawing_the_frequency_spectra():
    global filtered_image
    if len(filtered_image.shape) == 3:
        f = np.fft.fft2(cv2.cvtColor(filtered_image, cv2.COLOR_RGB2GRAY))
    else:
        f = np.fft.fft2(filtered_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    Freq_plot.imshow(magnitude_spectrum, cmap=plt.get_cmap('gray'))
    Freq_Figure.canvas.draw_idle()
    Original_Figure.canvas.draw_idle()


thread_1 = threading.Thread(target=drawing_the_frequency_spectra)
thread_2 = threading.Thread(target=draw_filtered_image)
thread_3 = threading.Thread(target=draw_histogram)

root = tk.Tk()
# frames intailization

# frame that contains button , combobox ,etc.
ToolBar_Frame = Frame(root)

# frame that will conatin all images , original and filterd
Images_Frame = Frame(root)
OriginalImage_Frame = Frame(Images_Frame)
FilteredImage_Frame = Frame(Images_Frame)

# frame that will contain frequency domain and histogram graph
Freq_and_Histo_Frame = Frame(root)
Freq_domain_Frame = Frame(Freq_and_Histo_Frame)
Histogram_Frame = Frame(Freq_and_Histo_Frame)

# packing original three frames to display in screen
ToolBar_Frame.pack(side=TOP, fill='both')
Images_Frame.pack(side=TOP, expand=True, fill='both')
Freq_and_Histo_Frame.pack(side=TOP, expand=True, fill='both')

# packing subframes for each graph
OriginalImage_Frame.pack(side=LEFT, expand=True, fill='both')
FilteredImage_Frame.pack(side=LEFT, expand=True, fill='both')
Freq_domain_Frame.pack(side=LEFT, expand=True, fill='both')
Histogram_Frame.pack(side=LEFT, expand=True, fill='both')

# creating toolbar elements and packing them , button
Open_Image_Button = tk.Button(ToolBar_Frame, text='open', command=openImage, padx='5', pady='5')
Open_Image_Button.pack(side=LEFT)

# combobox
Filters_list = tk.StringVar()
filters = ['Low pass freq domain', 'High pass freq domain', 'Low pass spatial domain', 'high pass spatial domain',
           'Median', 'Laplacian', 'H Equalization']
Filters_CB = ttk.Combobox(ToolBar_Frame, textvariable=Filters_list, font="Verdana 16 bold",
                          width=20)
Filters_CB['values'] = filters
Filters_CB['state'] = 'readonly'
Filters_CB.pack(side=LEFT)
Filters_CB.bind("<<ComboboxSelected>>", ApplyFilter)

# check box
gray_scale = tk.IntVar()
R1 = Radiobutton(ToolBar_Frame, text="Gray  ", variable=gray_scale, value=1,
                 command=sel)
R2 = Radiobutton(ToolBar_Frame, text="colored ", variable=gray_scale, value=0,
                 command=sel2)

checkbox = tk.Checkbutton(ToolBar_Frame, text='Gray Scale', variable=gray_scale)
R1.pack(side=RIGHT)
R2.pack(side=RIGHT)

# variable names that will be used in creation of images
Original_Figure = 'placeholder1'
Original_plot = 'placeholder2'
Filtered_Figure = 'placeholder3'
Filterd_plot = 'placeholder4'
Freq_Figure = 'placeholder5'
Freq_plot = 'placehold'
Histogram_Figure = 'placer'
Histogram_plot = 'place'

# creating figueres using function above
Original_Figure, Original_plot = fig_creation(Original_Figure, Original_plot, OriginalImage_Frame)
Filtered_Figure, Filterd_plot = fig_creation(Filtered_Figure, Filterd_plot, FilteredImage_Frame)
Original_plot.get_xaxis().set_visible(False)
Original_plot.get_yaxis().set_visible(False)
Filterd_plot.get_xaxis().set_visible(False)
Filterd_plot.get_yaxis().set_visible(False)
Freq_Figure, Freq_plot = fig_creation(Freq_Figure, Freq_plot, Freq_domain_Frame)
Histogram_Figure, Histogram_plot = fig_creation(Histogram_Figure, Histogram_plot, Histogram_Frame)

# putting titles for each image
Original_plot.set_title('Original image')
Filterd_plot.set_title('Filtered image')
Freq_plot.set_title('Frequency domain')
Histogram_plot.set_title('Histogram')

# coloring for background
Original_Figure.patch.set_facecolor('#C5E3FA')
Filtered_Figure.patch.set_facecolor('#C5E3FA')
Freq_Figure.patch.set_facecolor('#C5E3FA')
Histogram_Figure.patch.set_facecolor('#C5E3FA')

root.mainloop()
