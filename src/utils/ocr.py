import os
import cv2
import pytesseract
from PIL import Image
from typing import Dict, List
from utils.cli import debug_print

def get_tmp_dir():
    """Create and return path to the temporary directory in current working directory"""
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

def perform_ocr(image_path, pre_processor="thresh"):
    """Extract text from image using OCR"""
    if not os.path.exists(image_path):
        debug_print(f"Error: Image file not found: {image_path}")
        return ""
        
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            debug_print(f"Error: Could not load image: {image_path}")
            return ""
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply preprocessing based on selected method
        if pre_processor == "thresh":
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif pre_processor == "blur":
            gray = cv2.medianBlur(gray, 3)

        # Save temporary processed image and extract text
        tmp_dir = get_tmp_dir()
        filename = os.path.join(tmp_dir, f"{os.getpid()}.jpg")
        cv2.imwrite(filename, gray)
        
        text = pytesseract.image_to_string(Image.open(filename))
        
        # Clean up temporary file
        if os.path.exists(filename):
            os.remove(filename)

        return text
    except Exception as e:
        debug_print(f"Error processing image {image_path}: {str(e)}")
        return ""

def process_screenshots_folder(folder_path: str, pre_processor: str) -> List[Dict]:
    """Process all screenshots in a folder and convert to data format"""
    screenshots_data = []
    
    if not os.path.exists(folder_path):
        debug_print(f"Error: Screenshot folder not found: {folder_path}")
        return screenshots_data
    
    # Get all image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        f for f in os.listdir(folder_path) if 
        os.path.isfile(os.path.join(folder_path, f)) and 
        any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    
    for img_file in image_files:
        file_path = os.path.join(folder_path, img_file)
        
        # Extract text using OCR
        ocr_text = perform_ocr(file_path, pre_processor)
        
        # Create screenshot data object
        screenshot_data = {
            "filename": img_file,
            "timestamp": os.path.getmtime(file_path),
            "ocr_text": ocr_text,
            "file_path": file_path
        }
        
        screenshots_data.append(screenshot_data)
    
    return screenshots_data

def process_single_screenshot(image_path: str, pre_processor: str) -> Dict:
    """Process a single screenshot and convert to data format"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Screenshot file not found: {image_path}")
        
    # Extract text using OCR
    ocr_text = perform_ocr(image_path, pre_processor)
    
    # Create screenshot data object
    return {
        "filename": os.path.basename(image_path),
        "timestamp": os.path.getmtime(image_path),
        "ocr_text": ocr_text,
        "file_path": image_path
    }
