from PIL import Image
import numpy as np
from math import ceil

# get a mask image from an input image using a threshold all pixels that exceed the threshold are included in the mask, all others excluded
def get_mask_from_image(image: Image, threshold: int=100) -> Image:
    fn = lambda x : 255 if x > threshold else 0
    mask_img = image.convert('L').point(fn, mode='1')
    return mask_img

# crop image maximally s.t. object fills cropped image completely
def crop_out(im: Image, threshold: int=245) -> Image:
    arr = np.array(im)
    # find all indices in the image that have values below the threshold i.e. are not white
    indices = np.where(arr < threshold)
    # determine minmal and maximal indices along each dimension i.e. a tight bounding box around the object
    try:
        left_most = ceil(np.min(indices[1])/2)*2
    except ValueError:
        print(indices)
    right_most = ceil(np.max(indices[1])/2)*2
    top_most = ceil(np.min(indices[0])/2)*2
    bottom_most = ceil(np.max(indices[0])/2)*2

    # ensure proper box
    if right_most - left_most <= 0:
        right_most = left_most + 2
    if bottom_most-top_most <= 0:
        bottom_most = top_most + 2

    # crop image maximally and return image
    cropped_arr = arr[top_most:bottom_most, left_most:right_most]
    return Image.fromarray(cropped_arr)

# scaling of object
def scale_img(cropped: Image, scaling_factor: float) -> Image:
    return cropped.resize((ceil((cropped.width*scaling_factor)/2)*2, ceil((cropped.height*scaling_factor)/2)*2), Image.Resampling.NEAREST)

# reference point of bike is center, reference point of image is top left
def paste(cropped: Image, original_image: Image, fraction_down: float, fraction_right: float, rescale: bool=False):
    # calculate new starting location of object
    down_pixels = ceil(fraction_down * original_image.height)
    right_pixels = ceil(fraction_right * original_image.width)

    # ensure object fits at position
    if down_pixels < cropped.height//2:
        raise AssertionError('Your bike is too close to the top to fit, either downscale or move down')
    if original_image.height - (original_image.height - down_pixels) < cropped.height//2:
        raise AssertionError('Your bike is too close to the bottom to fit, either downscale or move up')
    if right_pixels < cropped.width//2:
        raise AssertionError('Your bike is too close to the left to fit, either downscale or move right')
    if original_image.width - (original_image.width - right_pixels) < cropped.width//2:
        raise AssertionError('Your bike is too close to the right to fit, either downscale or move left')

    cropped_arr = np.array(cropped)
    cropped_height, cropped_width = cropped_arr.shape[0], cropped_arr.shape[1]
    empty = (np.ones_like(np.array(original_image))*255).astype(np.uint8)  # create pure white image
    # insert object at new location into pure white image
    empty[down_pixels-cropped_height//2:down_pixels+cropped_height//2, right_pixels-cropped_width//2:right_pixels+cropped_width//2] = cropped_arr
    image = Image.fromarray(empty)
    
    # optionally rescale image to have width and height 256
    if rescale:
        if image.width > image.height:
            empty_im = Image.new('RGB', (image.width, image.width), color='white')
        else:
            empty_im = Image.new('RGB', (image.height, image.height), color='white')
        empty_im.paste(image, (empty_im.width//2-image.width//2, empty_im.height//2-image.height//2))
        image = empty_im.resize((256, 256), Image.Resampling.BICUBIC)

    return image

# paste_pipeline repositions the object in its frame
def paste_pipeline(im: Image, scale: float=1, fraction_down: float=0.5, fraction_right: float=0.5, rescale: bool=False, rotation: float=0) -> Image:
    # first the object is cropped out of the full frame
    cropped = crop_out(im)
    # then the object is scaled down
    scaled = scale_img(cropped, scale)
    # the object is pasted to the new location
    pasted = paste(scaled, im, fraction_down, fraction_right, rescale)
    # finally the full image is rotated
    pasted = pasted.rotate(rotation, fillcolor='white')
    return pasted