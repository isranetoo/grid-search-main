import os
import base64
from io import BytesIO
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from collections import Counter
from scipy.ndimage import gaussian_filter, median_filter

# Set Tesseract path if on Windows
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def load_image(bytes_data=None, file_path=None, image=None):
    """Load image from different input types."""
    if image:
        return image
    elif file_path:
        return Image.open(file_path)
    elif bytes_data:
        if isinstance(bytes_data, str):
            bytes_data = base64.b64decode(bytes_data)
        return Image.open(BytesIO(bytes_data))
    raise ValueError("Provide bytes_data, file_path, or image.")

def enhance_image(img, contrast_factor=3.5, sharpen_factor=2.0, brightness_factor=1.2):
    """Enhance image for better OCR readability with additional parameters."""
    # Convert to grayscale
    img = img.convert('L')
    
    # Apply multiple enhancement techniques
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpen_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Try multiple adaptive thresholding parameters
    results = []
    
    # Standard adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results.append(thresh1)
    
    # Different block sizes
    thresh2 = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 9, 3
    )
    results.append(thresh2)
    
    thresh3 = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results.append(thresh3)
    
    # Apply additional processing to each result
    final_results = []
    for result in results:
        # Apply morphological operations to clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Try bilateral filtering to preserve edges
        bilateral = cv2.bilateralFilter(cleaned, 9, 75, 75)
        final_results.append(bilateral)
        
        # Try median filter for noise reduction
        median = median_filter(cleaned, size=2)
        final_results.append(median)
        
        # Try gaussian blur
        gaussian = gaussian_filter(cleaned, sigma=0.6)
        final_results.append(gaussian)
    
    # Return the original result for now
    # A more sophisticated approach would analyze all results and choose the best
    return Image.fromarray(final_results[0])

def remove_lines(img, horizontal_size=25, vertical_size=25):
    """Remove horizontal and vertical lines with configurable parameters."""
    img_array = np.array(img.convert('L'))
    
    # Create binary image
    _, binary = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0,0,0), 3)
    
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0,0,0), 3)
    
    # Clean up small noise with configurable parameters
    denoise_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, denoise_kernel)
    
    # Additional noise removal techniques
    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(255 - binary, None, 10, 7, 21)
    
    # Invert back
    final = cv2.bitwise_not(denoised)
    
    return Image.fromarray(final)

def isolate_characters(img, min_size=15):
    """Try to isolate individual characters with configurable parameters."""
    img_array = np.array(img.convert('L'))
    
    # Apply connected component analysis
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - img_array, connectivity=8
    )
    
    # Filter components by size to remove noise
    filtered_img = np.zeros_like(img_array)
    for i in range(1, ret):  # Skip background (i=0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_img[labels == i] = 255
    
    # Additional morphological closing to fill gaps within characters
    kernel = np.ones((2, 2), np.uint8)
    filled = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Add edge enhancement
    edges = cv2.Canny(filled, 100, 200)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine original with edge information
    combined = cv2.addWeighted(filled, 0.7, dilated_edges, 0.3, 0)
    
    return Image.fromarray(combined)

def apply_ocr(img):
    """Apply OCR with different configurations."""
    configs = [
        '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
        '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
        '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
        '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
        # New configs with different PSM modes
        '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
        '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
        '--psm 9 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz',
    ]
    
    results = []
    for config in configs:
        text = pytesseract.image_to_string(img, config=config).strip()
        # Clean up the result
        text = ''.join(c for c in text if c.isalnum())
        if 3 <= len(text) <= 8:
            results.append(text)
    
    # If we have multiple results, choose the most common one
    if results:
        most_common = Counter(results).most_common(1)
        return most_common[0][0] if most_common else results[0]
    return ""

def solve_captcha(bytes_data=None, file_path=None, image=None):
    """Main function to process and extract text from captcha."""
    try:
        img = load_image(bytes_data, file_path, image)
        
        # Apply enhanced preprocessing pipeline
        img_no_lines = remove_lines(img)
        enhanced_img = enhance_image(img_no_lines)
        isolated_chars = isolate_characters(enhanced_img)
        
        # Try OCR on both the enhanced image and isolated characters
        result1 = apply_ocr(enhanced_img)
        result2 = apply_ocr(isolated_chars)
        
        # If both results match, we have higher confidence
        if result1 == result2 and result1:
            return result1
        
        # Otherwise, choose the better result
        if len(result1) >= 3 and all(c.isalnum() for c in result1):
            return result1
        if len(result2) >= 3 and all(c.isalnum() for c in result2):
            return result2
        
        # Fallback
        return result1 or result2
        
    except Exception as e:
        print(f"Error processing captcha: {str(e)}")
        return ""