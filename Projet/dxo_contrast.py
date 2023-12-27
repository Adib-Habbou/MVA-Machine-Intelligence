import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def adjust_contrast(image, contrast_level):
    """Adjust the image contrast and apply a gamma correction for tone mapping."""
    # Adjust the contrast using the existing method
    f = 131 * (contrast_level + 127) / (127 * (131 - contrast_level))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    adjusted_image = cv2.addWeighted(image, alpha_c, np.zeros(image.shape, image.dtype), 0, gamma_c)
    
    return adjusted_image

def adjust_micro_contrast(image, amount):
    """Adjust the micro contrast of an image."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    micro_contrast_img = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return micro_contrast_img

def adjust_fine_contrast(image, amount):
    """Adjust the fine contrast of an image."""
    # Convert to LAB color space for more perceptually uniform contrast adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply a subtle contrast enhancement to the L channel
    clahe = cv2.createCLAHE(clipLimit=amount, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to RGB
    updated_lab = cv2.merge((l, a, b))
    fine_contrast_image = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
    return fine_contrast_image

def adjust_highlights(image, amount):
    """Adjust the highlights of an image."""
    # Convert the image to HSV for easier manipulation of brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Apply a curve to the V channel to adjust highlights
    increase = np.array([min(value + amount, 255) for value in range(256)], dtype=np.uint8)
    v_highlights = cv2.LUT(v, increase)

    # Merge back the channels and convert to RGB
    hsv_adjusted = cv2.merge([h, s, v_highlights])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    
    return adjusted_image

def adjust_midtones(image, amount):
    """Adjust the midtones of an image."""
    # Convert the image to HSV for easier manipulation of brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Apply a curve to the V channel to adjust midtones
    midpoint = 128
    v_midtones = np.interp(v, [0, midpoint, 255], [0, midpoint + amount, 255]).astype(np.uint8)

    # Ensure that v_midtones is reshaped correctly
    v_midtones = v_midtones.reshape(v.shape)

    # Merge back the channels and convert to RGB
    hsv_adjusted = cv2.merge([h, s, v_midtones])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    
    return adjusted_image

def adjust_shadows(image, amount):
    """Adjust the shadows of an image, focusing on enhancing darker areas."""
    # Convert the image to HSV for easier manipulation of brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Define a threshold below which the pixels are considered shadows
    shadow_threshold = 128

    # Scale the amount of adjustment to be more subtle
    scaled_amount = (amount / 100) * 0.5  # Scale the amount for subtler adjustment

    # Apply the adjustment to the V channel, focusing on shadows
    v = v.astype(np.float32)  # Convert to float for manipulation
    v_adjusted = np.where(v < shadow_threshold, v + (shadow_threshold - v) * scaled_amount, v)

    # Ensure v_adjusted is clipped to valid range and converted back to uint8
    v_adjusted = np.clip(v_adjusted, 0, 255).astype(np.uint8)

    # Merge back the channels and convert to RGB
    hsv_adjusted = cv2.merge([h, s, v_adjusted])
    adjusted_image = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    
    return adjusted_image


def update_image(value):
    """Apply all adjustments to the image based on current slider values."""
    global current_image

    # Apply all adjustments
    adjusted_image = adjust_contrast(image.copy(), float(contrast_slider.get()))
    adjusted_image = adjust_micro_contrast(adjusted_image, float(micro_contrast_slider.get()) / 50)
    adjusted_image = adjust_fine_contrast(adjusted_image, float(fine_contrast_slider.get()) / 100)
    adjusted_image = adjust_highlights(adjusted_image, int(highlights_slider.get()) // 15)
    adjusted_image = adjust_midtones(adjusted_image, int(midtones_slider.get()))
    adjusted_image = adjust_shadows(adjusted_image, -int(shadows_slider.get()) // 5)
    

    # Update image display
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(adjusted_image))
    image_label.config(image=tk_img)
    image_label.image = tk_img


def update_midtones(value):
    """ Update the image based on the midtones slider value """
    global current_image

    # Convert the slider value from string to integer
    amount = int(value)

    # Apply midtones adjustment
    midtones_image = adjust_midtones(image.copy(), amount)

    # Update image display
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(midtones_image))
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Update histogram display (if applicable)
    # update_histogram(midtones_image)

def update_highlights(value):
    """ Update the image based on the highlights slider value """
    global current_image

    # Convert the slider value from string to integer
    amount = int(value) // 15

    # Apply highlights adjustment
    highlights_image = adjust_highlights(image.copy(), amount)

    # Update image display
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(highlights_image))
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Update histogram display (if applicable)
    # update_histogram(highlights_image)

def update_fine_contrast(value):
    """ Update the fine contrast of the image based on the slider value """
    global current_image

    # Calculate the fine contrast adjustment amount
    amount = float(value)/100  
    # Apply fine contrast adjustment
    fine_contrast_image = adjust_fine_contrast(image.copy(), amount)

    # Update image display
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(fine_contrast_image))
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Update histogram display (if you have one)
    # update_histogram(fine_contrast_image)


def update_micro_contrast(value):
    """ Update the micro contrast of the image based on the slider value """
    global current_image

    # Calculate the micro contrast adjustment amount
    amount = float(value) / 50  # Assuming the slider range is from 0 to 100

    # Apply micro contrast adjustment
    micro_contrast_image = adjust_micro_contrast(image.copy(), amount)

    # Update image display
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(micro_contrast_image))
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Update histogram display (if you have one)
    # update_histogram(micro_contrast_image)

def update_contrast(value):
    """ Update the contrast of the image """
    global current_image

    contrast_img = adjust_contrast(image.copy(), float(value))

    # Update the global current image
    current_image = contrast_img

    # Update image display
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(contrast_img))
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Update histogram display
    # update_histogram(contrast_img)

# def update_histogram(img):
#     """ Update the histogram plot including the luminance channel """
#     fig.clear()

#     # Plot the histogram for each color channel
#     color = ('r', 'g', 'b')
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         plt.plot(histr, color=col)

#     # Convert to grayscale to represent luminance
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     hist_luminance = cv2.calcHist([gray], [0], None, [256], [0, 256])

#     # Plot the luminance histogram
#     plt.plot(hist_luminance, color='k', linestyle='--')  # 'k' stands for black color
#     plt.title('Histogram for RGB and Luminance')
#     plt.xlabel('Pixel Value')
#     plt.ylabel('Frequency')
#     plt.xlim([0, 256])

#     canvas.draw()

# Load your image using OpenCV
image_path = 'Images/Portrait.JPG'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
current_image = image.copy()  # Copy of the image for contrast adjustments

# Resize the image for larger display (adjust the scale factor as needed)
scale_percent = 6  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Create a Tkinter window
root = Tk()
root.title("DXO Retro Engineering Contrast")

# Change the logo image below
root.iconbitmap('dxo_logo.ico')

# Load the logo image
logo_image = Image.open('dxo.png')  # Update this path to your logo image file
logo_image = logo_image.resize((50, 25), Image.ANTIALIAS)  # Resize logo if needed
tk_logo_image = ImageTk.PhotoImage(logo_image)

# Create a label for the logo image
logo_label = Label(root, image=tk_logo_image)
logo_label.image = tk_logo_image  # Keep a reference
logo_label.pack(side='bottom', anchor='sw')  # Position the logo at the bottom left


# Create a Label to display the image
tk_img = ImageTk.PhotoImage(image=Image.fromarray(image))
image_label = Label(root, image=tk_img)
image_label.pack()

# Create a slider for contrast adjustment
contrast_slider = Scale(root, from_=-25, to=25, orient=HORIZONTAL, command=update_image, label="Contrast")
contrast_slider.set(0)  # set the default value of slider
contrast_slider.pack()

# Create a slider for micro contrast adjustment
micro_contrast_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Micro Contrast", command=update_image)
micro_contrast_slider.set(0)  # set the default value of slider
micro_contrast_slider.pack()

# Create a slider for fine contrast adjustment
fine_contrast_slider = Scale(root, from_=1, to=100, orient=HORIZONTAL, label="Fine Contrast", command=update_image)
fine_contrast_slider.set(0)  # set the default value of slider
fine_contrast_slider.pack()

highlights_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Highlights", command=update_image)
highlights_slider.set(0)  # set the default value of slider
highlights_slider.pack()

# Create a slider for midtones adjustment
midtones_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Midtones", command=update_image)
midtones_slider.set(0)  # set the default value of slider
midtones_slider.pack()

shadows_slider = Scale(root, from_=-100, to=100, orient=HORIZONTAL, label="Shadows", command=update_image)
shadows_slider.set(0)  # set the default value of slider
shadows_slider.pack()

# Create a figure for the histogram
# fig, ax = plt.subplots()
# canvas = FigureCanvasTkAgg(fig, master=root)
# widget = canvas.get_tk_widget()
# widget.pack()

# Initial histogram display
# update_histogram(image)

# Run the application
root.mainloop()
