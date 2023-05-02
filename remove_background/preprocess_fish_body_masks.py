import os
import shutil   
from PIL import Image


def split_masks_and_images(folder):
    if not os.path.exists('body_masks/'):
        os.makedirs('body_masks/')
    for image_name in os.listdir(folder):
            if 'mask' in image_name:
                image_path = folder + "/" + image_name
                print(image_path)
                shutil.move(image_path, 'body_masks/')

def binarize_fish_body_masks(folder):
    for image_name in os.listdir(folder):
        image = Image.open(folder + "/" + image_name)
        image = image.convert("L")
        # Get the pixel data and iterate over each pixel
        pixels = image.load()
        for x in range(image.width):
            for y in range(image.height):
                # Get the pixel value
                if pixels[x, y] == 1:
                    pixels[x, y] = 255
                else:
                    pixels[x, y] = 0

        # Save the resulting binary image
        image.save(folder + "/" + image_name)
               
#binarize_fish_body_masks('body_masks/')

def remove_background(original_image, mask_image):

    #original_image = Image.open('C:/Users/ingvilrh/master_data/fish_bodies/IMG_0882_JPG.rf.efc7e53dcd35a1028e265f9a6e028f2d.jpg')
    #mask_image = Image.open('body_masks/IMG_0882_JPG.rf.efc7e53dcd35a1028e265f9a6e028f2d_mask.png')

    # Convert the mask image to a 1-bit (black and white) image
    mask_image = mask_image.convert("1")

    # Create a new image with a transparent alpha channel
    result_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))

    # Paste the original image onto the new image
    result_image.paste(original_image, (0, 0))

    # Add the mask image to the alpha channel of the new image
    result_image.putalpha(mask_image)

    # Save the resulting image
    result_image.save("result_image.png")