from diffusers import StableDiffusionPipeline
import torch
import os

def main():
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # prompt = "Create a high-resolution, photorealistic image of a birdfeeder. The birdfeeder is clear, filled with a mix of seeds. A small colorful bird with vibrant yellow, black, and white plumage is perched calmly on one side of the feeder. On the other side, a squirrel is clinging to the feeder."
    guidance_scale = 7.5  # Adjust this value as needed
    prompt = "A photograph of a high-resolution, photorealistic image of a bird and a squirrel at a birdfeeder. A colorful bird is snacking calmly from one side of the feeder. On the other side, a small brown-gray squirrel is clinging to the feeder, highest quality possible"

    image_dir = './mySampleImages'
    num_images = 5  # Number of images to generate
    for i in range(num_images):
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        image = pipe(prompt=prompt, guidance_scale=guidance_scale).images[0]
        image.save(os.path.join(image_dir, f"generated_{i}.png"))

if __name__ == "__main__":
    main()