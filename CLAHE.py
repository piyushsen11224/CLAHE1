import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def apply_clahe_color(input_folder, output_folder, metrics_folder, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Create folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        # Read the input image in color
        input_image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(input_image_path)

        # Split the image into individual channels
        channels = cv2.split(img)

        # Apply CLAHE to each channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_channels = [clahe.apply(channel) for channel in channels]

        # Merge the channels back to form the color image
        clahe_img = cv2.merge(clahe_channels)

        # Save the output image to the specified folder
        output_path = os.path.join(output_folder, f"clahe_output_{image_file}")
        cv2.imwrite(output_path, clahe_img)

        # Calculate metrics (MSE, PSNR, SSIM) for color images
        original_img = cv2.imread(input_image_path)
        mse = np.mean((original_img - clahe_img) ** 2)
        psnr = cv2.PSNR(original_img, clahe_img)
        ssim_value = np.mean([ssim(original_img[:,:,i], clahe_img[:,:,i]) for i in range(original_img.shape[2])])

        # Save metrics to a file
        metrics_output_path = os.path.join(metrics_folder, f"metrics_{image_file.replace('.', '_')}.txt")
        with open(metrics_output_path, 'w') as file:
            file.write(f'MSE: {mse}\n')
            file.write(f'PSNR: {psnr}\n')
            file.write(f'SSIM: {ssim_value}\n')

        print(f"CLAHE applied successfully. Output image saved at: {output_path}")
        print(f"Metrics saved at: {metrics_output_path}")
        print("---------------------------------------------")

# Example usage
input_folder = "/Users/piyush/Downloads/raw-890"
output_folder = "/Users/piyush/Desktop/output_images"
metrics_folder = "/Users/piyush/Desktop/metrixdata"

apply_clahe_color(input_folder, output_folder, metrics_folder)
