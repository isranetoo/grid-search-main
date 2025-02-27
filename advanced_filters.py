import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter, median_filter, binary_fill_holes
import skimage.morphology as morphology
from skimage.segmentation import clear_border
from collections import Counter

def adaptive_threshold(img_array, block_size=15, c=5):
    """Apply adaptive thresholding to handle varying lighting conditions."""
    # Ensure the image is grayscale
    if len(img_array.shape) > 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive threshold
    return cv2.adaptiveThreshold(
        img_array,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )

def noise_removal(img_array, kernel_size=3):
    """Remove noise using median filtering and morphological operations."""
    # Apply median filter to remove salt-and-pepper noise
    filtered = median_filter(img_array, size=kernel_size)
    
    # Apply morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed

def line_removal(img_array):
    """Advanced method to remove horizontal and vertical lines."""
    # Create a binary image
    if img_array.max() > 1:
        binary = img_array.copy()
    else:
        binary = (img_array * 255).astype(np.uint8)
    
    # Invert to make lines more visible
    inverted = cv2.bitwise_not(binary)
    
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_contours = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_contours = horizontal_contours[0] if len(horizontal_contours) == 2 else horizontal_contours[1]
    
    # Draw horizontal contours on original image to remove lines
    for c in horizontal_contours:
        cv2.drawContours(binary, [c], -1, (255, 255, 255), 3)
    
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_contours = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_contours = vertical_contours[0] if len(vertical_contours) == 2 else vertical_contours[1]
    
    # Draw vertical contours on original image to remove lines
    for c in vertical_contours:
        cv2.drawContours(binary, [c], -1, (255, 255, 255), 3)
    
    # Clean up remaining noise
    clean_img = noise_removal(binary)
    
    return clean_img

def character_isolation(img_array):
    """Isolate individual characters in the image."""
    # Ensure binary image
    if img_array.dtype != np.bool:
        binary = img_array > 128
    else:
        binary = img_array
    
    # Fill small holes
    filled = binary_fill_holes(binary)
    
    # Label connected components
    labeled, num_features = morphology.label(filled, return_num=True)
    
    # Filter by size to remove small noise
    component_sizes = np.bincount(labeled.ravel())
    too_small = component_sizes < 15
    too_small_mask = too_small[labeled]
    labeled[too_small_mask] = 0
    
    # Remove components touching border
    cleaned = clear_border(labeled)
    
    # Create final binary image
    result = cleaned > 0
    
    return result.astype(np.uint8) * 255

def enhance_contrast(img):
    """Enhance image contrast using multiple techniques."""
    # Convert to PIL Image if needed
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Ensure grayscale
    img = img.convert('L')
    
    # Apply contrast enhancement
    enhanced = ImageEnhance.Contrast(img).enhance(2.5)
    
    # Apply brightness enhancement
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.2)
    
    # Apply sharpening
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    
    # Apply additional filters
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    
    return enhanced

def deskew_image(img_array):
    """Deskew (straighten) image if it's tilted."""
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Calculate rotated bounding box
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) <= 0:
        return img_array
    
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Skip if angle is negligible
    if abs(angle) < 0.5:
        return img_array
    
    # Rotate the image
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img_array, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated

def advanced_preprocess(image, th1=185, th2=150, sigma1=1.0, sigma2=1.0):
    """Process image with advanced filtering techniques."""
    # Convert to numpy array if not already
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('L'))
    else:
        img_array = image
        
    # Step 1: Deskew the image
    deskewed = deskew_image(img_array)
    
    # Step 2: Apply adaptive thresholding
    thresholded = adaptive_threshold(deskewed, block_size=15, c=th1)
    
    # Step 3: Remove lines
    no_lines = line_removal(thresholded)
    
    # Step 4: Apply first Gaussian blur
    blurred = gaussian_filter(no_lines, sigma=sigma1)
    
    # Step 5: Apply second threshold
    _, second_threshold = cv2.threshold(
        blurred.astype(np.uint8), th2, 255, cv2.THRESH_BINARY
    )
    
    # Step 6: Remove noise
    clean = noise_removal(second_threshold)
    
    # Step 7: Apply second Gaussian blur for smoothing
    final_blurred = gaussian_filter(clean, sigma=sigma2)
    
    # Step 8: Final thresholding and morphological operations
    final = cv2.threshold(
        final_blurred.astype(np.uint8), th2, 255, cv2.THRESH_BINARY
    )[1]
    
    # Step 9: Attempt to isolate characters
    char_isolate = character_isolation(final)
    
    # Step 10: Convert back to PIL and enhance contrast
    final_image = enhance_contrast(Image.fromarray(char_isolate))
    
    return final_image

def try_multiple_filters(image):
    """Try multiple filtering strategies and return the best result."""
    # Start with some promising parameter combinations
    filter_params = [
        # th1, th2, sigma1, sigma2
        (185, 150, 1.0, 1.0),
        (175, 140, 1.1, 1.2),
        (195, 155, 0.9, 1.1),
        (180, 145, 1.2, 0.9),
    ]
    
    results = []
    for params in filter_params:
        processed = advanced_preprocess(image, *params)
        results.append(processed)
    
    return results

def preprocess_pipeline(image):
    """Complete preprocessing pipeline that tries multiple strategies."""
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Original enhanced image
    enhanced_original = enhance_contrast(image)
    
    # Try multiple filter strategies
    filter_results = try_multiple_filters(image)
    
    # Add enhanced original