from PIL import Image

# combine two PIL images by horizontal concatenation
def concat_PIL_h(im1, im2):
    if im1.height < im2.height:
        factor = im2.height/im1.height
        im1 = im1.resize((int(im1.width*factor), int(im1.height*factor)), Image.BICUBIC)
        
    if im2.height < im1.height:
        factor = im1.height/im2.height
        im2 = im2.resize((int(im2.width*factor), int(im2.height*factor)), Image.BICUBIC)
    
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# combine two PIL images by vertical concatenation
def concat_PIL_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst