import os
from transformers import pipeline


def subSelectImages(image_paths):

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


    selected_images = []
    for file in image_paths:
    # image = Image.open(image_path).convert("RGB")
        output = image_to_text(file)
        result = output[0]['generated_text']
        if ('bird' in result) and ('bird feeder' in result):
            selected_images.append(file)

    return selected_images

def main():
    # Path to the directory containing the images
    image_dir = 'images'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    selected_images = subSelectImages(image_paths)
    print(selected_images)

if __name__ == '__main__':
    main()