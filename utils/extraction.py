from PIL import Image
from utils import get_pasted_image, paste_pipeline

# -- some utilities to paste object into new images --

# this function is used to paste an object on pure white background into a given background image
def paste_image(fg: Image, bg: Image, fuzziness: int=18) -> Image:
    fg = fg.copy()
    bg = bg.copy()

    if fg.width < fg.height:
        factor = bg.height / fg.height
    else:
        factor = bg.width / fg.width
    fg = fg.resize((int(factor*fg.width), int(factor*fg.height)), Image.Resampling.BILINEAR)

    fg = fg.convert('RGBA')  # convert to RGBA to add alpha channel
    
    # replace white with tranparent
    pixdata = fg.load()
    # select the background color, this is typically white but could also be black or any other color
    bg_color = pixdata[0, 0]
    # iterate over all pixels in the foreground image
    for x in range(fg.size[0]):
        for y in range(fg.size[1]):
            # all pixels are to be replaced unless they are sufficiently (indicated by fuzziness) different from the background (i.e. first) pixel
            should_replace = True
            for idx in range(3):
                if abs(pixdata[x, y][idx] - bg_color[idx]) > fuzziness:
                    should_replace = False
            if should_replace:
                # all pixels to be replaced are replaced with (black) transpararent pixels
                pixdata[x, y] = (0, 0, 0, 0)
    
    # the modified transparent foreground is pasted onto the background
    bg.paste(fg, (0, 0), fg)
    return bg

# extract_object extracts an object from its original background and pastes it onto a white background
def extract_object(img: Image, bg: Image, object_desc: str, erosion_strength: int=3) -> Image:
    # match resolution of foreground to resolution of background
    if img.height > img.width:
        factor = bg.height / img.height
        img = img.resize((int(img.width*factor), int(img.height*factor)), Image.Resampling.LANCZOS)
    else:
        factor = bg.width / img.width
        img = img.resize((int(img.width*factor), int(img.height*factor)), Image.Resampling.LANCZOS)

    # use langSAM to extract object
    img = get_pasted_image(img, object_desc, erosion_strength=erosion_strength)
    # center object in frame
    img = paste_pipeline(img, 1, 0.5, 0.5)
    # create new all white background and paste object in the middle of it
    img_white = Image.new('RGB', (bg.width, bg.height), color='white')
    img_white.paste(img, (img_white.width//2-img.width//2, img_white.height//2-img.height//2))
    return img_white