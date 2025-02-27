import os
import base64
from io import BytesIO
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from collections import Counter

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

def enhance_image(img):
    """Enhance image for better OCR readability."""
    # Convert to grayscale
    img = img.convert('L')
    
    # Apply multiple enhancement techniques
    img = ImageEnhance.Contrast(img).enhance(3.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Apply adaptive thresholding for better character separation
    img_array = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((1, 1), np.uint8)
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return img

def remove_lines(img):
    """Remove horizontal and vertical lines using morphological operations."""
    img_array = np.array(img.convert('L'))
    
    # Create binary image
    _, binary = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0,0,0), 3)
    
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(binary, [c], -1, (0,0,0), 3)
    
    # Clean up small noise
    denoise_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, denoise_kernel)
    
    # Invert back
    binary = cv2.bitwise_not(binary)
    
    return Image.fromarray(binary)

def isolate_characters(img):
    """Try to isolate individual characters for better recognition."""
    img_array = np.array(img.convert('L'))
    
    # Apply connected component analysis
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - img_array, connectivity=8
    )
    
    # Filter components by size to remove noise
    min_size = 15
    filtered_img = np.zeros_like(img_array)
    for i in range(1, ret):  # Skip background (i=0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_img[labels == i] = 255
            
    return Image.fromarray(filtered_img)

def apply_ocr(img):
    """Apply OCR with different configurations."""
    configs = [
        '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789',
        '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789',
        '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789',
        '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789',
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