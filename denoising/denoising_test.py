import numpy as np
import cv2
from denoising_script import denoise_image

# Load test image
img = cv2.imread("test_image.png")

# Add noise to the image
noisy_img = img + np.random.normal(0, 20, img.shape).astype(np.uint8)

# Apply denoising
denoised_img = denoise_image(noisy_img)

# Save the denoised image
cv2.imwrite("denoised_image.png", denoised_img)

# Check if the PSNR (Peak Signal-to-Noise Ratio) between the original and denoised image is improved
psnr1 = cv2.PSNR(img, noisy_img)
psnr2 = cv2.PSNR(img, denoised_img)
print("PSNR (noisy image):", psnr1)
print("PSNR (denoised image):", psnr2)
if psnr2 > psnr1:
    print("Denoising successful!")
else:
    print("Denoising failed.")
