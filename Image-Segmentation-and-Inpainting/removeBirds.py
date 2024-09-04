import PIL
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import matplotlib.pyplot as plt
import os
from subSelectImages import subSelectImages


def removeBirds(bird_images, mask_paths):

    pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                          torch_dtype = torch.float16,)


    pipeline = pipeline.to("cuda")


    prompt = "photograph of a beautiful empty scene, highest quality possible"

    for file in bird_images:
        try:
            init_image = PIL.Image.open(file).convert("RGB")
            #Larger Mask Image file name ends with -larger-mask.png
            larger_mask_image = PIL.Image.open(file.rpartition(".")[0] + "-larger-mask.png").convert("RGB")

        except:
            print("Error: Could not open file")
            continue


        #Pipeling expects height and weight to be divisble by 8

        height = init_image.size[1]
        width = init_image.size[0]
        new_height = int(height/8)*8
        new_width = int(width/8)*8

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        init_image = init_image.crop((left, top, right, bottom))
        larger_mask_image = larger_mask_image.crop((left, top, right, bottom))

        #Run the pipeline
        image = pipeline(prompt=prompt, image=init_image, mask_image=larger_mask_image, height=new_height, width=new_width).images[0]
        image.save(file.rpartition(".")[0] + "-birdsRemoved.jpg")

def main():
    image_dir = 'images'

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    selected_images = subSelectImages(image_paths)
    mask_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith("-larger-mask.png")]


    removeBirds(selected_images, mask_paths)


if __name__ == '__main__':
    main()


