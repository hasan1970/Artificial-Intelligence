# Image Segmentation and Inpainting

This project involves selecting, segmenting, and modifying images using advanced image inpainting techniques. The core tasks include selecting images with specific objects (birds and birdfeeders), removing objects (birds), replacing objects (birds with other birds or squirrels), and generating new images using text prompts.

## Technical Overview

### Key Processes:
1. **Image Selection**:
   - Using a custom method to select 5 out of 20 images that contain birds and birdfeeders from a pool of images.
   
2. **Image Segmentation**:
   - Segmenting the birds in selected images to create masks for use in subsequent inpainting tasks.

3. **Image Inpainting**:
   - Inpainting is used to remove birds, replace them with other birds, or substitute them with squirrels, creating realistic image manipulations.

4. **Text-Based Image Generation**:
   - A generative model conditions on input text to create entirely new images of birds and squirrels at a bird feeder.

### Models Used:
- **Stable Diffusion**: Used for image inpainting and generating new images based on text prompts.
- **HRNet Segmentation Model**: Used for segmenting the birds in images to create masks.

## Contents

### Code
1. **subSelectImages.py**: 
   - Defines a function `subSelectImages` that processes 20 original images and selects 5 images containing birdfeeders and birds. The names of the 5 selected files are returned.
   
2. **segmentImages.py**: 
   - Defines a function `segmentImages` that takes a list of image filenames and segments the birds in each image, saving the resulting masks as PNG files.

3. **removeBirds.py**: 
   - Takes 5 images and their corresponding bird masks and removes the birds using inpainting. The resulting images are saved with the suffix `-birdsRemoved.jpg`.

4. **replaceBirds.py**: 
   - Takes 5 images and their corresponding bird masks, and replaces the birds with other similar-sized birds using inpainting. The resulting images are saved with the suffix `-birdsReplaced.jpg`.

5. **substituteSquirrels.py**: 
   - Takes 5 images and their corresponding bird masks, replacing the birds with squirrels using inpainting. The results are saved with the suffix `-andBirdsReplaced.jpg`.

6. **generateBirdFeederImagesFromText.py**: 
   - Loads a generative model that conditions on input text to produce 5 new images of squirrels competing with birds at a birdfeeder.

## How to Run

1. **Install dependencies**:
   Make sure to install the necessary libraries for Stable Diffusion, segmentation, and image processing.
   ```bash
   pip install torch torchvision transformers opencv-python
   ```

2. **Run the Image Manipulation Scripts**:
    - Select images containing birds and birdfeeders:

    ```bash
    python subSelectImages.py    
    ```

    - Segment birds in the selected images:

    ```bash
    python segmentImages.py
    ```

    - Remove birds from images:

    ```bash
    python removeBirds.py
    ```

    - Replace birds with other birds:

    ```bash
    python replaceBirds.py
    ```

    - Substitute birds with squirrels:

    ```bash
    python substituteSquirrels.py
    ```

3. **Generate Images from Text**: 
    Generate 5 new images based on the provided text prompts using a Stable Diffusion model::

    ```bash
    python generateBirdFeederImagesFromText.py
    ```

## Conclusion

This project showcases the use of image segmentation and inpainting techniques for sophisticated image manipulation tasks. By leveraging the power of Stable Diffusion, it effectively removes, replaces, and generates objects in images. The work demonstrates strong proficiency in handling complex image editing tasks using generative models.
