#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import cv2
from paddleocr import PPStructure
import time
import langdetect

# Constants
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

# Initialize the PP-Structure model with layout analysis
# Using PP-StructureV3 for better layout analysis
print("Initializing PP-Structure model...")
table_engine = PPStructure(layout=True, show_log=False, structure_version='PP-StructureV3')
print("Model initialized.")

def get_font_info(pdf_path, page_num, bbox):
    """Extract font information from a specific region in a PDF page."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Convert bbox from [x1, y1, x2, y2] to (x0, y0, x1, y1)
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Get text blocks with font info
    blocks = page.get_text("dict")["blocks"]
    
    # Store font sizes, weights, colors, and font names for text in the region
    font_sizes = []
    is_bold = False
    font_names = []
    colors = []
    
    for block in blocks:
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                # Check if span is within the bbox
                span_rect = fitz.Rect(span["bbox"])
                region_rect = fitz.Rect(x0, y0, x1, y1)
                
                if region_rect.intersects(span_rect):
                    font_sizes.append(span["size"])
                    # Check if font is bold
                    if "bold" in span["font"].lower():
                        is_bold = True
                    # Store font name and color
                    font_names.append(span["font"])
                    colors.append(span["color"])
    
    # Calculate average font size if any fonts found
    avg_font_size = np.mean(font_sizes) if font_sizes else 0
    
    # Get most common font name
    most_common_font = max(set(font_names), key=font_names.count) if font_names else ""
    
    # Get most common color
    most_common_color = max(set(colors), key=colors.count) if colors else 0
    
    doc.close()
    return {
        "avg_font_size": avg_font_size,
        "is_bold": is_bold,
        "font_name": most_common_font,
        "color": most_common_color
    }

def is_heading_by_pattern(text, language='en'):
    """
    Check if text matches common heading patterns.
    Supports multiple languages.
    """
    if language == 'en' or language.startswith('en-'):
        # English patterns
        # Check for numbered headings (e.g., 1., 1.1., 1.1.1.)
        if re.match(r'^\d+\.(\d+\.)*\s', text):
            return True
        
        # Check for alphabetic headings (e.g., A., a., i., I.)
        if re.match(r'^[A-Za-z]\.(\s|\d)', text):
            return True
        
        # Check for Roman numerals
        if re.match(r'^[IVXLCDM]+\.\s', text):
            return True
        
        # Check for common heading words
        heading_words = ['chapter', 'section', 'introduction', 'conclusion', 'appendix']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    elif language == 'ja' or language.startswith('ja-'):
        # Japanese patterns
        # Check for Japanese section markers (e.g., 第1章, 1.1節)
        if re.search(r'第\d+章|節', text):
            return True
        
        # Check for Japanese numbering patterns
        if re.match(r'^\d+\.\s', text) or re.match(r'^\(\d+\)', text):
            return True
    
    # Universal patterns that work across languages
    # Check for short text that's likely a heading (less than 50 chars, not ending with period)
    if len(text) < 50 and not text.endswith('.') and not text.endswith('。'):
        # Check if all caps or first letter of each word is capitalized
        if text.isupper() or text.title() == text:
            return True
    
    return False

def preprocess_image(img_np):
    """
    Preprocess the image to improve layout detection.
    - Deskewing
    - Contrast enhancement
    - Noise reduction
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Binarization using adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Deskew if needed
    # Find lines in the image
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    # If lines are found, calculate skew angle
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Only consider near-horizontal or near-vertical lines
            if (theta < 0.1 or abs(theta - np.pi/2) < 0.1 or abs(theta - np.pi) < 0.1):
                angles.append(theta)
        
        if angles:
            # Calculate median angle
            median_angle = np.median(angles)
            # Convert to degrees and normalize
            angle_degrees = np.degrees(median_angle)
            if angle_degrees > 45:
                angle_degrees = angle_degrees - 90
            
            # Only correct if skew is significant
            if abs(angle_degrees) > 0.5:
                # Get image dimensions
                h, w = binary.shape
                center = (w // 2, h // 2)
                # Create rotation matrix
                M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
                # Perform rotation
                binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
    
    # Convert back to RGB for the PP-Structure model
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return rgb

def detect_language(text):
    """
    Detect the language of the text.
    Returns language code (e.g., 'en', 'ja').
    """
    try:
        return langdetect.detect(text)
    except:
        return 'en'  # Default to English if detection fails

def create_feature_vector(block, mean_font_size, std_font_size, page_height, all_blocks):
    """
    Create a feature vector for heading classification.
    """
    bbox = block.get("bbox", [0, 0, 0, 0])
    text = block.get("text", "").strip()
    font_size = block.get("font_size", 0)
    is_bold = block.get("is_bold", False)
    is_title = block.get("is_title", False)
    font_name = block.get("font_name", "")
    color = block.get("color", 0)
    
    # Positional features
    x_position = bbox[0]  # Left position (indentation)
    y_position = bbox[1]  # Top position
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Text features
    text_length = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    has_numbering = 1 if re.match(r'^\d+\.(\d+\.)*\s', text) else 0
    all_caps = 1 if text.isupper() else 0
    title_case = 1 if text.title() == text else 0
    
    # Normalize features
    norm_font_size = (font_size - mean_font_size) / std_font_size if std_font_size > 0 else 0
    norm_x_position = x_position / 1000  # Assuming page width around 1000 pixels
    norm_y_position = y_position / page_height
    
    # Spacing features - check space above this block
    space_above = y_position
    for other_block in all_blocks:
        other_bbox = other_block.get("bbox", [0, 0, 0, 0])
        # If block is above the current block and horizontally overlapping
        if (other_bbox[3] <= y_position and 
            max(0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])) > 0):
            space_above = min(space_above, y_position - other_bbox[3])
    
    norm_space_above = space_above / 100  # Normalize by assuming 100 pixels is significant
    
    # Create feature vector
    features = [
        norm_font_size,
        1 if is_bold else 0,
        1 if is_title else 0,
        norm_x_position,
        norm_y_position,
        width / 1000,
        height / 100,
        text_length / 100,
        word_count / 10,
        avg_word_length / 10,
        has_numbering,
        all_caps,
        title_case,
        norm_space_above
    ]
    
    return features

def classify_heading_level(features, prev_heading_level=None):
    """
    Classify heading level based on features and previous heading level.
    Uses a rule-based approach that could be replaced with a trained classifier.
    """
    # Unpack features
    [norm_font_size, is_bold, is_title, norm_x_position, norm_y_position,
     width, height, text_length, word_count, avg_word_length,
     has_numbering, all_caps, title_case, norm_space_above] = features
    
    # Strong indicators for H1
    if is_title or norm_font_size > 1.5 or (all_caps and norm_font_size > 0.5):
        return "H1"
    
    # Strong indicators for H2
    if (norm_font_size > 0.75 or 
        (is_bold and norm_font_size > 0.25) or 
        (has_numbering and re.match(r'^\d+\.\s', text))):
        return "H2"
    
    # Strong indicators for H3
    if (norm_font_size > 0.25 or 
        (is_bold and norm_font_size > 0) or 
        (has_numbering and re.match(r'^\d+\.\d+\.\s', text))):
        return "H3"
    
    # Consider previous heading level for hierarchical consistency
    if prev_heading_level == "H1" and (norm_font_size > 0 or is_bold):
        return "H2"
    if prev_heading_level == "H2" and (norm_font_size > -0.5 or is_bold):
        return "H3"
    
    # Default case - not a heading
    return None

def classify_and_extract_headings(result, pdf_path, page_num, doc_language='en'):
    """
    Analyze the PP-Structure output and classify text blocks as headings.
    Returns a list of heading dictionaries.
    """
    headings = []
    
    # Skip if no result
    if not result:
        return headings
    
    # Get all text blocks
    text_blocks = []
    for item in result:
        if item.get("type") == "text":
            text_blocks.append(item)
    
    # If no text blocks, return empty list
    if not text_blocks:
        return headings
    
    # Get font information for all blocks
    for block in text_blocks:
        bbox = block.get("bbox", [0, 0, 0, 0])
        text = block.get("text", "").strip()
        
        # Skip empty text
        if not text:
            continue
        
        # Get font info
        font_info = get_font_info(pdf_path, page_num, bbox)
        block["font_size"] = font_info["avg_font_size"]
        block["is_bold"] = font_info["is_bold"]
        block["font_name"] = font_info["font_name"]
        block["color"] = font_info["color"]
        
        # Check if block is marked as title by PP-DocLayout
        block["is_title"] = block.get("type") == "title"
    
    # Calculate statistics for font sizes
    font_sizes = [block["font_size"] for block in text_blocks if block["font_size"] > 0]
    if not font_sizes:
        return headings
    
    mean_font_size = np.mean(font_sizes)
    std_font_size = np.std(font_sizes)
    
    # Get page height for normalization
    doc = fitz.open(pdf_path)
    page_height = doc[page_num].rect.height
    doc.close()
    
    # Sort blocks by vertical position (top to bottom)
    text_blocks.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
    
    # Process blocks in reading order
    prev_heading_level = None
    
    for block in text_blocks:
        text = block.get("text", "").strip()
        
        # Skip if no text
        if not text:
            continue
        
        # Create feature vector
        features = create_feature_vector(block, mean_font_size, std_font_size, page_height, text_blocks)
        
        # Classify heading level
        heading_level = classify_heading_level(features, prev_heading_level)
        
        # Additional check for heading patterns
        if not heading_level and is_heading_by_pattern(text, doc_language):
            # Use indentation and text pattern to determine level
            bbox = block.get("bbox", [0, 0, 0, 0])
            x_position = bbox[0]
            
            if x_position < 100:
                heading_level = "H1"
            elif x_position < 150:
                heading_level = "H2"
            else:
                heading_level = "H3"
        
        # If classified as a heading, add to the list and update previous level
        if heading_level:
            headings.append({
                "level": heading_level,
                "text": text,
                "page": page_num + 1  # 1-indexed page numbers
            })
            prev_heading_level = heading_level
    
    return headings

def extract_title(result, pdf_path):
    """
    Extract the document title from the first page.
    Uses multiple heuristics to identify the title.
    """
    if not result:
        return "Unknown Title"
    
    # Strategy 1: Use PP-DocLayout title type
    for item in result:
        if item.get("type") == "title":
            return item.get("text", "").strip()
    
    # Strategy 2: Use font size and position
    text_blocks = []
    for item in result:
        if item.get("type") == "text":
            bbox = item.get("bbox", [0, 0, 0, 0])
            text = item.get("text", "").strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Get font info
            font_info = get_font_info(pdf_path, 0, bbox)
            
            text_blocks.append({
                "text": text,
                "font_size": font_info["avg_font_size"],
                "is_bold": font_info["is_bold"],
                "y_position": bbox[1],  # Top position
                "x_position": bbox[0],  # Left position
                "width": bbox[2] - bbox[0],
                "is_centered": False
            })
    
    # If no text blocks, return default
    if not text_blocks:
        return "Unknown Title"
    
    # Calculate page width to determine if text is centered
    doc = fitz.open(pdf_path)
    page_width = doc[0].rect.width
    doc.close()
    
    # Check if blocks are centered
    for block in text_blocks:
        block_center = block["x_position"] + block["width"] / 2
        page_center = page_width / 2
        # If block center is within 10% of page center, consider it centered
        if abs(block_center - page_center) < (page_width * 0.1):
            block["is_centered"] = True
    
    # Sort by multiple criteria:
    # 1. Centered text (preferred for titles)
    # 2. Font size (larger first)
    # 3. Y-position (top first)
    text_blocks.sort(key=lambda x: (not x["is_centered"], -x["font_size"], x["y_position"]))
    
    # The largest font at the top of the page is likely the title
    for block in text_blocks:
        # Skip very short text as titles are usually longer
        if len(block["text"]) > 3 and len(block["text"]) < 100:  # Titles are rarely very long
            return block["text"]
    
    # Fallback to the first text block if nothing else works
    return text_blocks[0]["text"] if text_blocks else "Unknown Title"

def process_pdf(pdf_path):
    """
    Process a single PDF file and extract its outline.
    Returns a dictionary with title and outline.
    """
    print(f"Processing: {pdf_path}")
    start_time = time.time()
    
    # Initialize the outline
    outline = []
    title = "Unknown Title"
    doc_language = 'en'  # Default language
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Process each page
        for page_num in range(len(doc)):
            # Convert PDF page to image
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Preprocess the image
            preprocessed_img = preprocess_image(img_np)
            
            # Process the image with PP-Structure
            result = table_engine(preprocessed_img)
            
            # Extract title from the first page
            if page_num == 0:
                title = extract_title(result, pdf_path)
                # Detect document language from first page text
                all_text = " ".join([item.get("text", "") for item in result if item.get("type") == "text"])
                if all_text:
                    doc_language = detect_language(all_text)
            
            # Extract headings from the page
            page_headings = classify_and_extract_headings(result, pdf_path, page_num, doc_language)
            outline.extend(page_headings)
        
        doc.close()
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return {"title": title, "outline": outline}
    
    # Sort outline by page number
    outline.sort(key=lambda x: x["page"])
    
    # Post-process the outline to ensure hierarchical consistency
    processed_outline = []
    current_levels = {"H1": None, "H2": None, "H3": None}
    
    for item in outline:
        level = item["level"]
        
        # Update the current level
        current_levels[level] = item
        
        # Reset lower levels when a higher level is encountered
        if level == "H1":
            current_levels["H2"] = None
            current_levels["H3"] = None
        elif level == "H2":
            current_levels["H3"] = None
        
        processed_outline.append(item)
    
    print(f"Processed in {time.time() - start_time:.2f} seconds")
    return {"title": title, "outline": processed_outline}

def main():
    """Main function to process all PDFs in the input directory."""
    print(f"Starting PDF outline extraction")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = []
    for entry in os.scandir(INPUT_DIR):
        if entry.is_file() and entry.name.lower().endswith('.pdf'):
            pdf_files.append(entry.path)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        try:
            # Get the base filename
            base_name = os.path.basename(pdf_path)
            output_name = os.path.splitext(base_name)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            # Process the PDF
            result = process_pdf(pdf_path)
            
            # Write the result to a JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Output written to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    print("PDF outline extraction complete")

if __name__ == "__main__":
    main() 