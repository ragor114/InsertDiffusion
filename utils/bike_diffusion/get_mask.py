import pandas as pd
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.patches import Ellipse
import math
import numpy as np

RELEVANT_COLUMNS = ["x_rear_wheel_center", "y_rear_wheel_center", 
        "x_BB", "y_BB", 
        "x_front_wheel_center","y_front_wheel_center", 
        "x_head_tube_top", "y_head_tube_top", 
        "x_rear_tube_connect_seat_tube", "y_rear_tube_connect_seat_tube", 
        "x_top_tube_connect_seat_tube", "y_top_tube_connect_seat_tube",
        "x_top_tube_connect_head_tube", "y_top_tube_connect_head_tube",
        "x_down_tube_connect_head_tube", "y_down_tube_connect_head_tube", 
        "x_stem_top", "y_stem_top", 
        "x_front_fork", "y_front_fork",
        "x_saddle_top", "y_saddle_top", 
        "x_seat_tube_top", "y_seat_tube_top", "Bike index",
        "Wheel diameter front", "Wheel diameter rear", "Teeth on chainring"]

# utility to generate PIL image from matplotlib figure
def fig2img(fig: figure) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# based on the data in the csv this function creates a mask by appropriately connecting reference points and extruding the lines
# further masks for wheels are created
def row2figure(row) -> figure:
    face_color = 'black'
    face_color_wheel = 'black'

    fig = plt.figure()

    ax = plt.axes()
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('None')

    # plt.figure(figsize=(25, 20), dpi = 100)
    ax.set_xlim(-200, 3500) 
    ax.set_ylim(-500, 3500)

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    x_1 = row['x_rear_wheel_center'] + 10
    y_1 = row['y_rear_wheel_center'] + 10
    plt.plot(x_1, y_1, marker='o', markersize=2, color=face_color)

    x_2 = row['x_BB'] + 10
    y_2 = row['y_BB'] + 10
    plt.plot(x_2, y_2, marker='o', markersize=2, color=face_color)

    x_3 = row['x_front_wheel_center'] + 10
    y_3 = row['y_front_wheel_center'] + 10
    plt.plot(x_3, y_3, marker='o', markersize=2, color=face_color)

    x_4 = row['x_head_tube_top'] + 10
    y_4 = row['y_head_tube_top'] + 10
    plt.plot(x_4, y_4, marker='o', markersize=2, color=face_color)

    x_5 = row['x_rear_tube_connect_seat_tube'] + 10
    y_5 = row['y_rear_tube_connect_seat_tube'] + 10
    # For the tripe joint experiment for mask 394 seed 2000
    # plt.plot(x_5, y_5, marker='o', markersize=2, color='w', zorder = 200)

    x_6 = row['x_top_tube_connect_seat_tube'] + 10
    y_6 = row['y_top_tube_connect_seat_tube'] + 10
    plt.plot(x_6, y_6, marker='o', markersize=2, color=face_color)

    x_7 = row['x_top_tube_connect_head_tube'] + 10
    y_7 = row['y_top_tube_connect_head_tube'] + 10
    # plt.plot(x_7, y_7, marker='o', markersize=2, color='w', zorder=202)

    # x_57 = (x_5 + x_7) / 2
    # y_57 = (y_5 + y_7) / 2
    # plt.plot(x_57, y_57, marker='o', markersize=2, color='w', zorder=202)

    x_8 = row['x_down_tube_connect_head_tube'] + 10
    y_8 = row['y_down_tube_connect_head_tube'] + 10
    plt.plot(x_8, y_8, marker='o', markersize=2, color=face_color)

    x_9 = row['x_stem_top'] + 10
    y_9 = row['y_stem_top'] + 10
    plt.plot(x_9, y_9, marker='o', markersize=2, color=face_color)

    x_10 = row['x_front_fork'] + 10
    y_10 = row['y_front_fork'] + 10
    plt.plot(x_10, y_10, marker='o', markersize=2, color=face_color)

    x_11 = row['x_saddle_top'] + 10
    y_11 = row['y_saddle_top'] + 10
    # Removed this for the new saddle test
    #plt.plot(x_11, y_11, marker='o', markersize=2, color=face_color)

    x_12 = row['x_seat_tube_top'] + 10
    y_12 = row['y_seat_tube_top'] + 10
    plt.plot(x_12, y_12, marker='o', markersize=2, color=face_color)

    # plot lines and cycles

    # setting the default widths for the circles and the lines
    line_width = 8
    circle_width = 2
    full_wheels = False

    if full_wheels:
        circle_rear = plt.Circle((x_1, y_1), row['Wheel diameter rear'] / 2, facecolor=face_color, edgecolor=face_color)
        ax.add_artist(circle_rear)
        circle_front = plt.Circle((x_3, y_3), row['Wheel diameter front'] / 2, facecolor=face_color, edgecolor=face_color)
        ax.add_artist(circle_front)
    else:
        plt.plot(x_1 - row['Wheel diameter rear'] / 2, y_1, marker='o', markersize=.5, color=face_color)
        plt.plot(x_1, y_1 - row['Wheel diameter rear'] / 2, marker='o', markersize=.5, color=face_color)
        


    circle_rog = plt.Circle((x_2, y_2), row['Teeth on chainring'] * 2, facecolor=face_color, edgecolor=face_color)
    ax.add_artist(circle_rog)

    plt.plot([x_1, x_2], [y_1, y_2], color=face_color, linewidth=row['Teeth on chainring'] / math.sqrt(2) , solid_capstyle='round')
    plt.plot([x_1, x_5], [y_1, y_5], color=face_color, linewidth=line_width, solid_capstyle='round')
    plt.plot([x_2, x_8], [y_2, y_8], color=face_color, linewidth=line_width, solid_capstyle='round')
    plt.plot([x_10, x_3], [y_10, y_3], color=face_color, linewidth=line_width, solid_capstyle='round')
    plt.plot([x_8, x_7], [y_8, y_7], color=face_color, linewidth=line_width, solid_capstyle='round')

    if y_7 <= y_8:
        plt.plot([x_7, x_8], [y_7, y_8], color=face_color, linewidth=line_width, solid_capstyle='round')
        plt.plot([x_10, x_4], [y_10, y_4], color=face_color, linewidth=line_width, solid_capstyle='round')
    else:
        plt.plot([x_7, x_4], [y_7, y_4], color=face_color, linewidth=line_width, solid_capstyle='round')
        plt.plot([x_10, x_8], [y_10, y_8], color=face_color, linewidth=line_width, solid_capstyle='round')

    plt.plot([x_9, x_4], [y_9, y_4], color=face_color, linewidth=line_width, solid_capstyle='round')
    plt.plot([x_6, x_7], [y_6, y_7], color=face_color, linewidth=line_width, solid_capstyle='round')
    
    # Calculate the angle of the top bar
    saddle_angle = math.atan((y_7 - y_6) / (x_7 - x_6)) * 180 / math.pi

    points = [(x_12, y_12), (x_5, y_5), (x_6, y_6)]
    max_y_point = max(points, key=lambda point: point[1])
    # plt.plot([x_11, max_y_point[0], x_12], [y_11, max_y_point[1], y_12], color=face_color, linewidth=line_width, solid_capstyle='round')
    # plt.plot([x_11, x_12], [y_11, y_12], color=face_color)

    plt.plot([x_6, x_5], [y_6, y_5], color=face_color, linewidth=line_width, solid_capstyle='round')
    
    x_saddle = 0
    y_saddle = 0

    if y_6 <= y_5:
        plt.plot([x_2 - (x_2-x_6) * 1.35, x_2], [1.35*y_6,y_2], color=face_color, linewidth=line_width, solid_capstyle='round')
        x_saddle = x_2 - (x_2-x_6) * 1.35
        y_saddle = 1.35*y_6
        # plt.plot([x_6, x_2], [y_6, y_2], color=face_color, linewidth=line_width, solid_capstyle='round')
        # plt.plot([x_5, x_12], [y_5, y_12], color=face_color, linewidth=line_width, solid_capstyle='round')
    else:
        
        # New Method for the saddle point
        plt.plot([x_2 - (x_2-x_5) * 1.35, x_2], [1.35*y_5,y_2], color=face_color, linewidth=line_width, solid_capstyle='round')
        x_saddle = x_2 - (x_2-x_5) * 1.35
        y_saddle = 1.35*y_5
        # plt.plot([x_5, x_2], [y_5, y_2], color=face_color, linewidth=line_width, solid_capstyle='round')
        # plt.plot([x_6, x_12], [y_6, y_12], color=face_color, linewidth=line_width, solid_capstyle='round')

    # Plot other circular areas
    # Gears
    circle = plt.Circle((x_1, y_3), 80, facecolor=face_color, edgecolor=face_color)
    ax.add_artist(circle)
    # Front Brakes
    circle = plt.Circle((x_3, y_3), 80, facecolor=face_color, edgecolor=face_color)
    ax.add_artist(circle)
    # Pedals
    circle = plt.Circle((x_2, y_2), 120, facecolor=face_color, edgecolor=face_color)
    ax.add_artist(circle)
    # Oval Handle
    height = 250
    ellipse = Ellipse((x_9 + 25, y_9), width=450, height=height, angle=-30, facecolor=face_color, edgecolor=face_color)
    ax.add_artist(ellipse)
    # Oval Saddle
    height = 250
    ellipse = Ellipse((x_saddle, y_saddle - height/2), width=400, height=height, angle=saddle_angle, facecolor=face_color, edgecolor=face_color)
    ax.add_artist(ellipse)

    # Removing a part of the mask in order to add a trijoint
    # circle = plt.Circle((x_5, y_5), 50, facecolor='w', edgecolor='w', zorder=100)

    ax.add_artist(circle)

    plt.axis('off')

    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.close()

    return fig

# this function crops the mask image maximally to accomodate the full mask and returns the cropped mask as a numpy array
def get_cropped_mask_array(mask_image: Image, threshold: int=200) -> np.array:
    # create image with only black and white pixels
    fn = lambda x : 255 if x > threshold else 0
    img = mask_image.convert('L').point(fn, mode='1')
    
    # create array
    array = np.array(img)
    array = 1 - array
    
    # cropping
    x, y = np.nonzero(array)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    
    array = array[xl:xr+1, yl:yr+1]
    array = (1 - array) * 255
    
    # placement in 256x256 image
    height, length = array.shape[0], array.shape[1]
    lower_margin_length = 68
    
    upper_margin_length = 256 - height - lower_margin_length
    horizontal_margin = 256 - length
    upper_margin_length = max (0, upper_margin_length)
    horizontal_margin = max (0, horizontal_margin)
    length = max(0, length)
    
    upper_margin_pad = np.full((upper_margin_length, length), 255)
    lower_margin_pad = np.full((lower_margin_length, length), 255)
    right_margin_pad = np.full((256, horizontal_margin//2+10), 255)
    left_margin_pad = np.full((256, horizontal_margin//2-10), 255)
    
    array = np.hstack((left_margin_pad, np.hstack((np.vstack((upper_margin_pad, array, lower_margin_pad)), right_margin_pad))))
    return array

# retrieves the appropriate dataframe row based on the index provided
def get_dataframe_row(parameter_csv_path: str, row_idx: int, restrict_relevant_columns: bool=False):
    df = pd.read_csv(parameter_csv_path)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    if restrict_relevant_columns:
        df = df[RELEVANT_COLUMNS]
    return df[df['Bike index'] == row_idx].iloc[0]

# this method generates the mask based on the csv and idx provided
def get_mask(parameter_csv_path: str, index_of_bike: int=0) -> Image:
    if parameter_csv_path.split('.')[-1] != 'csv':
        raise ValueError('The path you provided does not point to a csv file.')

    # get dataframe row corresponding to idx
    row = get_dataframe_row(parameter_csv_path, index_of_bike, True)

    # get plt figure with mask
    fig = row2figure(row)
    img = fig2img(fig)
    # crop mask maximally and turn into image
    img = Image.fromarray(np.uint8(get_cropped_mask_array(img)), mode='L')

    # resize cropped mask to full size such that mask fills image completely
    # TODO: there might be a better solution for this problem
    # sometimes the mask does not match the expected image dimension by a few pixels
    img = img.resize((256, 256), Image.Resampling.BICUBIC)
    return img

# this function is used to replace the background of the wheel images with a new color typically pure white
def clean_image_background(img: Image, desired_color: tuple, fuzziness: int=5):
    pixdata = img.load()
    wheel_background_color = pixdata[0, 0]
    fuzziness = 5
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            shouldReplace = True
            for idx in range(3): #include RGB but exclude alpha
                if abs(pixdata[x, y][idx] - wheel_background_color[idx]) > fuzziness:
                    shouldReplace = False
            if shouldReplace:
                pixdata[x, y] = desired_color
    return img

# this function inserts the given wheel at the appropriate location in a clean white image
def get_background_with_wheel(design_type: int, parameter_csv_path: str, bike_idx: int=0, width: int=256, height: int=256) -> Image:
    if design_type != 0 and design_type != 1:
        raise AssertionError('Only 0 and 1 are allowed as the wheel design type')
    scale = 0.118
    lower_margin_length = 68
    designs = ["bmx", "normal"]
    
    row = get_dataframe_row(parameter_csv_path, bike_idx)
    
    x_1 = row['x_rear_wheel_center'] + 10
    y_1 = row['y_rear_wheel_center'] + 10
    x_3 = row['x_front_wheel_center'] + 10
    y_3 = row['y_front_wheel_center'] + 10
    
    background = Image.new('RGB', (width, height), color='white')
    rear_wheel_image = Image.open(f"./utils/bike_diffusion/wheel_designs/{designs[design_type]}.png")
    rear_wheel_image = clean_image_background(rear_wheel_image, (255, 255, 255, 255))
    rear_wheel_size = row['Wheel diameter rear']
    newsize = (int(rear_wheel_size * scale), int(rear_wheel_size * scale))
    rear_wheel_image = rear_wheel_image.resize(newsize)
    
    front_wheel_image = Image.open(f"./utils/bike_diffusion/wheel_designs/{designs[design_type]}.png")
    front_wheel_image = clean_image_background(front_wheel_image, (255, 255, 255, 255))
    front_wheel_size = row['Wheel diameter front']
    newsize = (int(front_wheel_size * scale), int(front_wheel_size * scale))
    front_wheel_image = front_wheel_image.resize(newsize)
    
    rear_wheel_coordiantes = (int((x_1 - rear_wheel_size // 2) * scale - 3) + 76//2 - 10, 256 - int((y_1 + rear_wheel_size // 2) * scale + lower_margin_length))
    front_wheel_coordiantes = (int((x_3 - front_wheel_size // 2) * scale - 3) + 8 + 76//2 - 10, 256 - int((y_3 + rear_wheel_size // 2) * scale + lower_margin_length))
    
    background.paste(rear_wheel_image, rear_wheel_coordiantes)
    background.paste(front_wheel_image, front_wheel_coordiantes)
    return background

# wrapper to get mask and background with wheels at once
def get_mask_and_background(parameter_csv_path: str, bike_idx: int=0, design_type: int=0, width: int=256, height: int=256) -> tuple:
    return get_mask(parameter_csv_path, bike_idx), get_background_with_wheel(design_type, parameter_csv_path, bike_idx, width, height)
