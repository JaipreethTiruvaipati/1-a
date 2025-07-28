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
# Using PP-StructureV2 for better layout analysis
print("Initializing PP-Structure model...")
table_engine = PPStructure(layout=True, show_log=False, structure_version='PP-StructureV2')
print("Model initialized.")

def get_font_info(pdf_path, page_num, bbox):
    """Extract font information from a specific region in a PDF page."""
    try:
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
    except Exception as e:
        print(f"Warning: Error extracting font info: {e}")
        # Return default values if an error occurs
        avg_font_size = 0
        is_bold = False
        most_common_font = ""
        most_common_color = 0
    finally:
        if 'doc' in locals():
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
    Supports multiple languages including:
    - English (en)
    - Japanese (ja)
    - Spanish (es)
    - French (fr)
    - German (de)
    - Chinese (zh)
    - Arabic (ar)
    - Russian (ru)
    - Hindi (hi)
    - Portuguese (pt)
    - Italian (it)
    - Dutch (nl)
    - Korean (ko)
    - Turkish (tr)
    - Polish (pl)
    - Swedish (sv)
    - Danish (da)
    - Norwegian (no)
    - Finnish (fi)
    - Vietnamese (vi)
    - Thai (th)
    """
    # Skip empty or very short text
    if not text or len(text.strip()) < 2:
        return False
    
    # Common numeric patterns across languages
    # Check for numbered headings (e.g., 1., 1.1., 1.1.1.)
    if re.match(r'^\d+\.(\d+\.)*\s', text):
        return True
        
    # Check for parenthesized numbers (e.g., (1), (1.2))
    if re.match(r'^\(\d+(\.\d+)*\)', text):
        return True
    
    # Check for Roman numerals - common in many languages
    if re.match(r'^[IVXLCDM]+\.\s', text) or re.match(r'^[IVXLCDM]+\s', text):
        return True
    
    # Check for bullet points and other list markers
    if re.match(r'^[•※⚫⚪◦○●◉◎■□▪▫★☆♦♣♠♥➤➢➡⇒→-]\s', text):
        return True
        
    # Language-specific patterns
    lang_prefix = language.split('-')[0] if '-' in language else language
    
    # English patterns
    if lang_prefix == 'en':
        # Check for alphabetic headings (e.g., A., a., i., I.)
        if re.match(r'^[A-Za-z]\.(\s|\d)', text):
            return True
        
        # Check for common English heading words
        heading_words = ['chapter', 'section', 'introduction', 'conclusion', 'appendix', 'part', 
                         'summary', 'abstract', 'overview', 'preface', 'foreword', 'glossary']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # Japanese patterns
    elif lang_prefix == 'ja':
        # Check for Japanese section markers (e.g., 第1章, 1.1節, はじめに)
        if re.search(r'第\d+章|節|はじめに|まとめ|概要|要約', text):
            return True
    
    # Spanish patterns
    elif lang_prefix == 'es':
        heading_words = ['capítulo', 'sección', 'parte', 'introducción', 'conclusión', 
                         'resumen', 'apéndice', 'prólogo', 'prefacio', 'glosario']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # French patterns
    elif lang_prefix == 'fr':
        heading_words = ['chapitre', 'section', 'partie', 'introduction', 'conclusion', 
                        'résumé', 'annexe', 'préface', 'avant-propos', 'glossaire']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # German patterns
    elif lang_prefix == 'de':
        heading_words = ['kapitel', 'abschnitt', 'teil', 'einleitung', 'zusammenfassung',
                         'anhang', 'vorwort', 'glossar', 'überblick', 'einführung']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # Chinese patterns
    elif lang_prefix == 'zh':
        # Check for Chinese section markers (e.g., 第1章, 第一节, 引言, 总结)
        if re.search(r'第[一二三四五六七八九十百千万\d]+[章节篇部分]|引言|简介|摘要|总结|附录|概述', text):
            return True
    
    # Arabic patterns
    elif lang_prefix == 'ar':
        # Check for Arabic section markers and heading words
        if re.search(r'الفصل|القسم|الجزء|المقدمة|الخاتمة|الملخص|الملحق|تمهيد|مقدمة|خلاصة', text):
            return True
    
    # Russian patterns
    elif lang_prefix == 'ru':
        heading_words = ['глава', 'раздел', 'часть', 'введение', 'заключение', 
                         'аннотация', 'приложение', 'предисловие', 'резюме', 'обзор']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # Hindi patterns
    elif lang_prefix == 'hi':
        heading_words = ['अध्याय', 'खंड', 'भाग', 'परिचय', 'निष्कर्ष', 'सारांश', 'परिशिष्ट']
        if any(text.startswith(word) for word in heading_words):
            return True
    
    # Portuguese patterns
    elif lang_prefix == 'pt':
        heading_words = ['capítulo', 'seção', 'parte', 'introdução', 'conclusão', 
                         'resumo', 'apêndice', 'prefácio', 'glossário']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # Italian patterns
    elif lang_prefix == 'it':
        heading_words = ['capitolo', 'sezione', 'parte', 'introduzione', 'conclusione', 
                         'riassunto', 'appendice', 'prefazione', 'glossario']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
            
    # Korean patterns
    elif lang_prefix == 'ko':
        if re.search(r'제\s*\d+\s*장|서론|결론|요약|부록|개요|소개', text):
            return True
            
    # Turkish patterns
    elif lang_prefix == 'tr':
        heading_words = ['bölüm', 'kısım', 'giriş', 'sonuç', 'özet', 'ek', 'önsöz']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # Thai patterns
    elif lang_prefix == 'th':
        if re.search(r'บทที่|ส่วนที่|บทนำ|สรุป|ภาคผนวก|บทคัดย่อ', text):
            return True
            
    # Vietnamese patterns
    elif lang_prefix == 'vi':
        heading_words = ['chương', 'phần', 'mục', 'giới thiệu', 'kết luận', 'tóm tắt', 'phụ lục']
        if any(text.lower().startswith(word) for word in heading_words):
            return True
    
    # Universal patterns that work across languages
    
    # Check for section numbering with various separators (1-1, 1_1, etc.)
    if re.match(r'^\d+[\.-_]\d+', text):
        return True
        
    # Check for short text that's likely a heading (less than 50 chars, not ending with sentence-ending punctuation)
    if len(text) < 50:
        # Check for common sentence-ending punctuation across languages
        sentence_endings = ['.', '。', '؟', '!', '?', '።', '۔', '។', '၊', '።', '។', '።']
        if not any(text.rstrip().endswith(ending) for ending in sentence_endings):
            # Check for text formatting that suggests a heading
            if text.isupper() or text.title() == text:
                return True
            
            # Check if text starts with a capital letter and is short (likely a heading)
            if len(text) < 30 and text[0].isupper():
                return True
            
            # Check for indented or centered text (first character is space)
            if text.startswith(' ') and len(text.strip()) < 40:
                return True
                
            # Check for unusual whitespace that might indicate heading
            if text.count(' ') < 2 and len(text) < 25:
                return True
    
    # Check for text that ends with a colon (often indicates a heading)
    if text.rstrip().endswith(':') and len(text) < 60:
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
    Detect the language of the text with improved multi-language support.
    Returns language code (e.g., 'en', 'ja', 'zh', 'ar', etc.).
    Uses a more robust approach with multiple fallbacks.
    """
    if not text or len(text.strip()) < 5:
        return 'en'  # Default to English if text is too short
    
    # Try multiple samples from the text to improve accuracy
    try:
        # Clean the text - remove numbers, special chars and excessive whitespace
        clean_text = re.sub(r'\d+', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) > 50:
            # Take samples from different parts of the document
            samples = [
                clean_text[:100],  # Start
                clean_text[len(clean_text)//2:len(clean_text)//2+100],  # Middle
                clean_text[-100:]  # End
            ]
            
            # Detect language for each sample
            langs = []
            for sample in samples:
                try:
                    if len(sample.strip()) > 10:  # Only consider samples with enough text
                        detected_lang = langdetect.detect(sample)
                        # Filter out unreliable detections
                        if detected_lang not in ['un', 'unknown']:
                            langs.append(detected_lang)
                except:
                    continue
            
            # Return the most common language detected
            if langs:
                from collections import Counter
                most_common_lang = Counter(langs).most_common(1)[0][0]
                print(f"Detected language: {most_common_lang} (from multiple samples)")
                return most_common_lang
        
        # If we couldn't use multiple samples, try with the whole text
        lang = langdetect.detect(clean_text)
        if lang not in ['un', 'unknown']:
            print(f"Detected language: {lang}")
            return lang
    except Exception as e:
        print(f"Language detection failed: {e}")
    
    # Enhanced fallback detection based on character sets
    try:
        # Check for specific character sets that indicate language
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):  # Hiragana/Katakana
            return 'ja'
        elif re.search(r'[\u4E00-\u9FFF]', text):  # Chinese characters
            return 'zh'
        elif re.search(r'[\uAC00-\uD7AF]', text):  # Korean Hangul
            return 'ko'
        elif re.search(r'[\u0600-\u06FF]', text):  # Arabic
            return 'ar'
        elif re.search(r'[\u0E00-\u0E7F]', text):  # Thai
            return 'th'
        elif re.search(r'[\u0900-\u097F]', text):  # Devanagari (Hindi, etc.)
            return 'hi'
        elif re.search(r'[\u0400-\u04FF]', text):  # Cyrillic (Russian, etc.)
            return 'ru'
        elif re.search(r'[\u0B80-\u0BFF]', text):  # Tamil
            return 'ta'
        elif re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
            return 'kn'
        elif re.search(r'[\u0D00-\u0D7F]', text):  # Malayalam
            return 'ml'
        elif re.search(r'[\u0D80-\u0DFF]', text):  # Sinhala
            return 'si'
        elif re.search(r'[\u0E80-\u0EFF]', text):  # Lao
            return 'lo'
        elif re.search(r'[\u0F00-\u0FFF]', text):  # Tibetan
            return 'bo'
        elif re.search(r'[\u1000-\u109F]', text):  # Myanmar
            return 'my'
        elif re.search(r'[\u1100-\u11FF]', text):  # Hangul Jamo
            return 'ko'
        elif re.search(r'[\u1200-\u137F]', text):  # Ethiopic
            return 'am'
        elif re.search(r'[\u1400-\u167F]', text):  # Unified Canadian Aboriginal Syllabics
            return 'cr'
        elif re.search(r'[\u1680-\u169F]', text):  # Ogham
            return 'ga'
        elif re.search(r'[\u16A0-\u16FF]', text):  # Runic
            return 'non'
        elif re.search(r'[\u1700-\u171F]', text):  # Tagalog
            return 'tl'
        elif re.search(r'[\u1720-\u173F]', text):  # Hanunoo
            return 'hnn'
        elif re.search(r'[\u1740-\u175F]', text):  # Buhid
            return 'bku'
        elif re.search(r'[\u1760-\u177F]', text):  # Tagbanwa
            return 'tbw'
        elif re.search(r'[\u1780-\u17FF]', text):  # Khmer
            return 'km'
        elif re.search(r'[\u1800-\u18AF]', text):  # Mongolian
            return 'mn'
        elif re.search(r'[\u1900-\u194F]', text):  # Limbu
            return 'lif'
        elif re.search(r'[\u1950-\u197F]', text):  # Tai Le
            return 'tdd'
        elif re.search(r'[\u1980-\u19DF]', text):  # New Tai Lue
            return 'khb'
        elif re.search(r'[\u19E0-\u19FF]', text):  # Khmer Symbols
            return 'km'
        elif re.search(r'[\u1A00-\u1A1F]', text):  # Buginese
            return 'bug'
        elif re.search(r'[\u1A20-\u1AAF]', text):  # Tai Tham
            return 'nod'
        elif re.search(r'[\u1AB0-\u1AFF]', text):  # Combining Diacritical Marks Extended
            return 'en'  # Could be various languages
        elif re.search(r'[\u1B00-\u1B7F]', text):  # Balinese
            return 'ban'
        elif re.search(r'[\u1B80-\u1BBF]', text):  # Sundanese
            return 'su'
        elif re.search(r'[\u1BC0-\u1BFF]', text):  # Batak
            return 'btk'
        elif re.search(r'[\u1C00-\u1C4F]', text):  # Lepcha
            return 'lep'
        elif re.search(r'[\u1C50-\u1C7F]', text):  # Ol Chiki
            return 'sat'
        elif re.search(r'[\u1C80-\u1C8F]', text):  # Cyrillic Extended-C
            return 'ru'
        elif re.search(r'[\u1C90-\u1CBF]', text):  # Georgian Extended
            return 'ka'
        elif re.search(r'[\u1CC0-\u1CCF]', text):  # Sundanese Supplement
            return 'su'
        elif re.search(r'[\u1CD0-\u1CFF]', text):  # Vedic Extensions
            return 'sa'
        elif re.search(r'[\u1D00-\u1D7F]', text):  # Phonetic Extensions
            return 'en'  # Could be various languages
        elif re.search(r'[\u1D80-\u1DBF]', text):  # Phonetic Extensions Supplement
            return 'en'  # Could be various languages
        elif re.search(r'[\u1DC0-\u1DFF]', text):  # Combining Diacritical Marks Supplement
            return 'en'  # Could be various languages
        elif re.search(r'[\u1E00-\u1EFF]', text):  # Latin Extended Additional
            return 'en'  # Could be various languages
        elif re.search(r'[\u1F00-\u1FFF]', text):  # Greek Extended
            return 'el'
        elif re.search(r'[\u2000-\u206F]', text):  # General Punctuation
            return 'en'  # Could be various languages
        elif re.search(r'[\u2070-\u209F]', text):  # Superscripts and Subscripts
            return 'en'  # Could be various languages
        elif re.search(r'[\u20A0-\u20CF]', text):  # Currency Symbols
            return 'en'  # Could be various languages
        elif re.search(r'[\u20D0-\u20FF]', text):  # Combining Diacritical Marks for Symbols
            return 'en'  # Could be various languages
        elif re.search(r'[\u2100-\u214F]', text):  # Letterlike Symbols
            return 'en'  # Could be various languages
        elif re.search(r'[\u2150-\u218F]', text):  # Number Forms
            return 'en'  # Could be various languages
        elif re.search(r'[\u2190-\u21FF]', text):  # Arrows
            return 'en'  # Could be various languages
        elif re.search(r'[\u2200-\u22FF]', text):  # Mathematical Operators
            return 'en'  # Could be various languages
        elif re.search(r'[\u2300-\u23FF]', text):  # Miscellaneous Technical
            return 'en'  # Could be various languages
        elif re.search(r'[\u2400-\u243F]', text):  # Control Pictures
            return 'en'  # Could be various languages
        elif re.search(r'[\u2440-\u245F]', text):  # Optical Character Recognition
            return 'en'  # Could be various languages
        elif re.search(r'[\u2460-\u24FF]', text):  # Enclosed Alphanumerics
            return 'en'  # Could be various languages
        elif re.search(r'[\u2500-\u257F]', text):  # Box Drawing
            return 'en'  # Could be various languages
        elif re.search(r'[\u2580-\u259F]', text):  # Block Elements
            return 'en'  # Could be various languages
        elif re.search(r'[\u25A0-\u25FF]', text):  # Geometric Shapes
            return 'en'  # Could be various languages
        elif re.search(r'[\u2600-\u26FF]', text):  # Miscellaneous Symbols
            return 'en'  # Could be various languages
        elif re.search(r'[\u2700-\u27BF]', text):  # Dingbats
            return 'en'  # Could be various languages
        elif re.search(r'[\u27C0-\u27EF]', text):  # Miscellaneous Mathematical Symbols-A
            return 'en'  # Could be various languages
        elif re.search(r'[\u27F0-\u27FF]', text):  # Supplemental Arrows-A
            return 'en'  # Could be various languages
        elif re.search(r'[\u2800-\u28FF]', text):  # Braille Patterns
            return 'en'  # Could be various languages
        elif re.search(r'[\u2900-\u297F]', text):  # Supplemental Arrows-B
            return 'en'  # Could be various languages
        elif re.search(r'[\u2980-\u29FF]', text):  # Miscellaneous Mathematical Symbols-B
            return 'en'  # Could be various languages
        elif re.search(r'[\u2A00-\u2AFF]', text):  # Supplemental Mathematical Operators
            return 'en'  # Could be various languages
        elif re.search(r'[\u2B00-\u2BFF]', text):  # Miscellaneous Symbols and Arrows
            return 'en'  # Could be various languages
        elif re.search(r'[\u2C00-\u2C5F]', text):  # Glagolitic
            return 'cu'
        elif re.search(r'[\u2C60-\u2C7F]', text):  # Latin Extended-C
            return 'en'  # Could be various languages
        elif re.search(r'[\u2C80-\u2CFF]', text):  # Coptic
            return 'cop'
        elif re.search(r'[\u2D00-\u2D2F]', text):  # Georgian Supplement
            return 'ka'
        elif re.search(r'[\u2D30-\u2D7F]', text):  # Tifinagh
            return 'ber'
        elif re.search(r'[\u2D80-\u2DDF]', text):  # Ethiopic Extended
            return 'am'
        elif re.search(r'[\u2DE0-\u2DFF]', text):  # Cyrillic Extended-A
            return 'ru'
        elif re.search(r'[\u2E00-\u2E7F]', text):  # Supplemental Punctuation
            return 'en'  # Could be various languages
        elif re.search(r'[\u2E80-\u2EFF]', text):  # CJK Radicals Supplement
            return 'zh'
        elif re.search(r'[\u2F00-\u2FDF]', text):  # Kangxi Radicals
            return 'zh'
        elif re.search(r'[\u2FF0-\u2FFF]', text):  # Ideographic Description Characters
            return 'zh'
        elif re.search(r'[\u3000-\u303F]', text):  # CJK Symbols and Punctuation
            return 'zh'
        elif re.search(r'[\u3040-\u309F]', text):  # Hiragana
            return 'ja'
        elif re.search(r'[\u30A0-\u30FF]', text):  # Katakana
            return 'ja'
        elif re.search(r'[\u3100-\u312F]', text):  # Bopomofo
            return 'zh'
        elif re.search(r'[\u3130-\u318F]', text):  # Hangul Compatibility Jamo
            return 'ko'
        elif re.search(r'[\u3190-\u319F]', text):  # Kanbun
            return 'ja'
        elif re.search(r'[\u31A0-\u31BF]', text):  # Bopomofo Extended
            return 'zh'
        elif re.search(r'[\u31C0-\u31EF]', text):  # CJK Strokes
            return 'zh'
        elif re.search(r'[\u31F0-\u31FF]', text):  # Katakana Phonetic Extensions
            return 'ja'
        elif re.search(r'[\u3200-\u32FF]', text):  # Enclosed CJK Letters and Months
            return 'zh'
        elif re.search(r'[\u3300-\u33FF]', text):  # CJK Compatibility
            return 'zh'
        elif re.search(r'[\u3400-\u4DBF]', text):  # CJK Unified Ideographs Extension A
            return 'zh'
        elif re.search(r'[\u4DC0-\u4DFF]', text):  # Yijing Hexagram Symbols
            return 'zh'
        elif re.search(r'[\u4E00-\u9FFF]', text):  # CJK Unified Ideographs
            return 'zh'
        elif re.search(r'[\uA000-\uA48F]', text):  # Yi Syllables
            return 'ii'
        elif re.search(r'[\uA490-\uA4CF]', text):  # Yi Radicals
            return 'ii'
        elif re.search(r'[\uA4D0-\uA4FF]', text):  # Lisu
            return 'lis'
        elif re.search(r'[\uA500-\uA63F]', text):  # Vai
            return 'vai'
        elif re.search(r'[\uA640-\uA69F]', text):  # Cyrillic Extended-B
            return 'ru'
        elif re.search(r'[\uA6A0-\uA6FF]', text):  # Bamum
            return 'bax'
        elif re.search(r'[\uA700-\uA71F]', text):  # Modifier Tone Letters
            return 'en'  # Could be various languages
        elif re.search(r'[\uA720-\uA7FF]', text):  # Latin Extended-D
            return 'en'  # Could be various languages
        elif re.search(r'[\uA800-\uA82F]', text):  # Syloti Nagri
            return 'syl'
        elif re.search(r'[\uA830-\uA83F]', text):  # Common Indic Number Forms
            return 'en'  # Could be various languages
        elif re.search(r'[\uA840-\uA87F]', text):  # Phags-pa
            return 'xal'
        elif re.search(r'[\uA880-\uA8DF]', text):  # Saurashtra
            return 'saz'
        elif re.search(r'[\uA8E0-\uA8FF]', text):  # Devanagari Extended
            return 'hi'
        elif re.search(r'[\uA900-\uA92F]', text):  # Kayah Li
            return 'eky'
        elif re.search(r'[\uA930-\uA95F]', text):  # Rejang
            return 'rej'
        elif re.search(r'[\uA960-\uA97F]', text):  # Hangul Jamo Extended-A
            return 'ko'
        elif re.search(r'[\uA980-\uA9DF]', text):  # Javanese
            return 'jv'
        elif re.search(r'[\uA9E0-\uA9FF]', text):  # Myanmar Extended-B
            return 'my'
        elif re.search(r'[\uAA00-\uAA5F]', text):  # Cham
            return 'cja'
        elif re.search(r'[\uAA60-\uAA7F]', text):  # Myanmar Extended-A
            return 'my'
        elif re.search(r'[\uAA80-\uAADF]', text):  # Tai Viet
            return 'blt'
        elif re.search(r'[\uAAE0-\uAAFF]', text):  # Meetei Mayek Extensions
            return 'mni'
        elif re.search(r'[\uAB00-\uAB2F]', text):  # Ethiopic Extended-A
            return 'am'
        elif re.search(r'[\uAB30-\uAB6F]', text):  # Latin Extended-E
            return 'en'  # Could be various languages
        elif re.search(r'[\uAB70-\uABBF]', text):  # Cherokee Supplement
            return 'chr'
        elif re.search(r'[\uABC0-\uABFF]', text):  # Meetei Mayek
            return 'mni'
        elif re.search(r'[\uAC00-\uD7AF]', text):  # Hangul Syllables
            return 'ko'
        elif re.search(r'[\uD7B0-\uD7FF]', text):  # Hangul Jamo Extended-B
            return 'ko'
        elif re.search(r'[\uD800-\uDB7F]', text):  # High Surrogates
            return 'en'  # Could be various languages
        elif re.search(r'[\uDB80-\uDBFF]', text):  # High Private Use Surrogates
            return 'en'  # Could be various languages
        elif re.search(r'[\uDC00-\uDFFF]', text):  # Low Surrogates
            return 'en'  # Could be various languages
        elif re.search(r'[\uE000-\uF8FF]', text):  # Private Use Area
            return 'en'  # Could be various languages
        elif re.search(r'[\uF900-\uFAFF]', text):  # CJK Compatibility Ideographs
            return 'zh'
        elif re.search(r'[\uFB00-\uFB4F]', text):  # Alphabetic Presentation Forms
            return 'en'  # Could be various languages
        elif re.search(r'[\uFB50-\uFDFF]', text):  # Arabic Presentation Forms-A
            return 'ar'
        elif re.search(r'[\uFE00-\uFE0F]', text):  # Variation Selectors
            return 'en'  # Could be various languages
        elif re.search(r'[\uFE10-\uFE1F]', text):  # Vertical Forms
            return 'zh'
        elif re.search(r'[\uFE20-\uFE2F]', text):  # Combining Half Marks
            return 'en'  # Could be various languages
        elif re.search(r'[\uFE30-\uFE4F]', text):  # CJK Compatibility Forms
            return 'zh'
        elif re.search(r'[\uFE50-\uFE6F]', text):  # Small Form Variants
            return 'zh'
        elif re.search(r'[\uFE70-\uFEFF]', text):  # Arabic Presentation Forms-B
            return 'ar'
        elif re.search(r'[\uFF00-\uFFEF]', text):  # Halfwidth and Fullwidth Forms
            return 'zh'
        elif re.search(r'[\uFFF0-\uFFFF]', text):  # Specials
            return 'en'  # Could be various languages
    except Exception as e:
        print(f"Character set detection failed: {e}")
    
    print(f"Language detection failed, defaulting to English")
    return 'en'  # Default to English if all detection methods fail

def create_feature_vector(block, mean_font_size, std_font_size, page_height, all_blocks):
    """
    Create a feature vector for heading classification.
    """
    try:
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
            try:
                other_bbox = other_block.get("bbox", [0, 0, 0, 0])
                # If block is above the current block and horizontally overlapping
                if (other_bbox[3] <= y_position and 
                    max(0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])) > 0):
                    space_above = min(space_above, y_position - other_bbox[3])
            except Exception:
                continue
        
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
    
    except Exception as e:
        print(f"Warning: Error creating feature vector: {e}")
        # Return default feature vector
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
        has_numbering):  # Simplified condition without text reference
        return "H2"
    
    # Strong indicators for H3
    if (norm_font_size > 0.25 or 
        (is_bold and norm_font_size > 0)):  # Simplified condition without text reference
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
        try:
            bbox = block.get("bbox", [0, 0, 0, 0])
            text = block.get("text", "").strip()
            
            # For empty text blocks, set a default font_size and skip.
            # This prevents a KeyError in the list comprehension below.
            if not text:
                block['font_size'] = 0
                continue
            
            # Get font info
            font_info = get_font_info(pdf_path, page_num, bbox)
            
            # Check if all required keys exist in font_info
            if "avg_font_size" not in font_info or "is_bold" not in font_info or "font_name" not in font_info or "color" not in font_info:
                # Use default values if missing
                block["font_size"] = 0
                block["is_bold"] = False
                block["font_name"] = ""
                block["color"] = 0
            else:
                block["font_size"] = font_info["avg_font_size"]
                block["is_bold"] = font_info["is_bold"]
                block["font_name"] = font_info["font_name"]
                block["color"] = font_info["color"]
            
            # Check if block is marked as title by PP-DocLayout
            block["is_title"] = block.get("type") == "title"
        except Exception as e:
            # If there's any error processing this block, set default values
            block["font_size"] = 0
            block["is_bold"] = False
            block["font_name"] = ""
            block["color"] = 0
            block["is_title"] = False
            print(f"Warning: Error processing block: {e}")
    
    # Calculate statistics for font sizes
    font_sizes = [block["font_size"] for block in text_blocks if block["font_size"] > 0]
    if not font_sizes:
        return headings
    
    mean_font_size = np.mean(font_sizes)
    std_font_size = np.std(font_sizes)
    
    # Get page height for normalization
    try:
        doc = fitz.open(pdf_path)
        page_height = doc[page_num].rect.height
        doc.close()
    except Exception as e:
        print(f"Warning: Error getting page height: {e}")
        page_height = 1000  # Default value if can't get real height
    
    # Sort blocks by vertical position (top to bottom)
    text_blocks.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
    
    # Process blocks in reading order
    prev_heading_level = None
    
    for block in text_blocks:
        try:
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
        except Exception as e:
            print(f"Warning: Error classifying heading: {e}")
            continue
    
    return headings

def extract_title(result, pdf_path):
    """
    Extract the document title from the first page.
    Uses multiple heuristics to identify the title.
    """
    if not result:
        # Try directly extracting title from PyMuPDF as fallback
        try:
            doc = fitz.open(pdf_path)
            # Get first page text
            first_page_text = doc[0].get_text()
            # Get metadata
            metadata_title = doc.metadata.get("title", "")
            doc.close()
            
            # If metadata has title, use it
            if metadata_title and len(metadata_title) > 3:
                return metadata_title
                
            # Otherwise extract first non-empty line from first page
            lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
            if lines:
                # Use first line that's not too long and not too short
                for line in lines:
                    if 3 < len(line) < 100:
                        return line
                # If no suitable line found, use first line
                if lines:
                    return lines[0]
        except Exception as e:
            print(f"Warning: Failed to extract fallback title: {e}")
        return "Unknown Title"
    
    # Get PDF metadata for title
    try:
        doc = fitz.open(pdf_path)
        if doc.metadata and doc.metadata.get('title'):
            title_from_metadata = doc.metadata.get('title').strip()
            if title_from_metadata:
                return title_from_metadata
        doc.close()
    except Exception as e:
        print(f"Warning: Error getting PDF metadata: {e}")
    
    # Strategy 1: Use PP-DocLayout title type
    for item in result:
        if item.get("type") == "title":
            title_text = item.get("text", "").strip()
            if title_text:  # Ensure title is not empty
                return title_text
    
    # Strategy 2: Use font size and position
    text_blocks = []
    try:
        for item in result:
            if item.get("type") == "text":
                bbox = item.get("bbox", [0, 0, 0, 0])
                text = item.get("text", "").strip()
                
                # Skip empty text
                if not text:
                    continue
                
                # Get font info
                try:
                    font_info = get_font_info(pdf_path, 0, bbox)
                    
                    text_blocks.append({
                        "text": text,
                        "font_size": font_info.get("avg_font_size", 0),
                        "is_bold": font_info.get("is_bold", False),
                        "y_position": bbox[1],  # Top position
                        "x_position": bbox[0],  # Left position
                        "width": bbox[2] - bbox[0],
                        "is_centered": False
                    })
                except Exception as e:
                    # If font info fails, still add the block with defaults
                    text_blocks.append({
                        "text": text,
                        "font_size": 0,
                        "is_bold": False,
                        "y_position": bbox[1],
                        "x_position": bbox[0],
                        "width": bbox[2] - bbox[0],
                        "is_centered": False
                    })
    except Exception as e:
        print(f"Warning: Error processing text blocks: {e}")
    
    # If no text blocks, try fallback approach
    if not text_blocks:
        # Strategy 3: Try direct text extraction
        try:
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                page_text = doc[0].get_text()
                doc.close()
                
                if page_text:
                    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                    if lines:
                        return lines[0][:100]  # First line, limited to 100 chars
        except Exception as e:
            print(f"Warning: Error with direct text extraction: {e}")
            
        # Extract filename as last resort
        try:
            base_filename = os.path.basename(pdf_path)
            title_from_filename = os.path.splitext(base_filename)[0]
            return title_from_filename
        except:
            return "Unknown Title"
    
    # Calculate page width to determine if text is centered
    try:
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
    except Exception as e:
        print(f"Warning: Error determining text centering: {e}")
    
    # Sort by multiple criteria:
    # 1. Centered text (preferred for titles)
    # 2. Font size (larger first)
    # 3. Y-position (top first)
    text_blocks.sort(key=lambda x: (not x.get("is_centered", False), -x.get("font_size", 0), x.get("y_position", 0)))
    
    # The largest font at the top of the page is likely the title
    for block in text_blocks:
        # Skip very short text as titles are usually longer
        if len(block["text"]) > 3 and len(block["text"]) < 100:  # Titles are rarely very long
            return block["text"]
    
    # Fallback to the first text block if nothing else works
    if text_blocks:
        return text_blocks[0]["text"]
    
    # Extract filename as last resort
    base_filename = os.path.basename(pdf_path)
    title_from_filename = os.path.splitext(base_filename)[0]
    return title_from_filename

def process_pdf(pdf_path, lang=None):
    """
    Process a single PDF file and extract its outline.
    Returns a dictionary with title and outline.
    
    Args:
        pdf_path (str): Path to the PDF file
        lang (str, optional): Language code if known. Defaults to None (will be detected).
    """
    print(f"Processing: {pdf_path}")
    start_time = time.time()
    
    # Initialize the outline
    outline = []
    title = "Unknown Title"
    
    # First check how many pages the PDF has
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")
        doc.close()
    except Exception as e:
        print(f"Warning: Could not determine page count: {e}")
        total_pages = 0
    
    # Detect language if not provided
    doc_language = lang if lang else 'en'  # Default to English
    
    # Extract filename as a basic title fallback
    base_filename = os.path.basename(pdf_path)
    filename_title = os.path.splitext(base_filename)[0]
    title = filename_title  # Start with filename as title
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Try to get title from metadata first
        metadata_title = doc.metadata.get("title", "")
        if metadata_title and len(metadata_title.strip()) > 3:
            title = metadata_title.strip()
        
        # Process each page
        for page_num in range(len(doc)):
            try:
                # First attempt: Direct text extraction with PyMuPDF
                page = doc[page_num]
                page_text = page.get_text("dict")
                
                # Fallback text extraction for title on first page
                if page_num == 0 and title == "Unknown Title":
                    # Get first few text blocks and check for potential title
                    blocks = page_text.get("blocks", [])
                    if blocks:
                        # Try to find title in first few text blocks
                        for i, block in enumerate(blocks[:3]):  # Check first 3 blocks
                            if "lines" in block:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        text = " ".join([span.get("text", "").strip() for span in line["spans"]])
                                        if text and 3 < len(text) < 100:
                                            title = text
                                            break
                
                # Extract language if not detected yet
                if page_num == 0:
                    try:
                        text_content = page.get_text()
                        if text_content:
                            doc_language = detect_language(text_content)
                    except:
                        pass
                
                # Analyze blocks directly from PyMuPDF for headings
                pymupdf_headings = []
                for block in page_text.get("blocks", []):
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if "spans" not in line:
                            continue
                            
                        # Get text and font info from spans
                        text = " ".join([span.get("text", "").strip() for span in line["spans"]])
                        if not text or len(text) > 100:  # Skip empty or very long text
                            continue
                            
                        # Get font attributes from first span (main span)
                        if line["spans"]:
                            main_span = line["spans"][0]
                            font_size = main_span.get("size", 0)
                            font_name = main_span.get("font", "")
                            is_bold = "bold" in font_name.lower()
                            
                            # Classify as heading based on font properties and text
                            heading_level = None
                            
                            # Simple heuristic: larger fonts are higher-level headings
                            if font_size > 14 or is_bold and font_size > 12:
                                heading_level = "H1"
                            elif font_size > 12 or is_bold and font_size > 10:
                                heading_level = "H2"
                            elif font_size > 10 or is_bold:
                                heading_level = "H3"
                            elif is_heading_by_pattern(text, doc_language):
                                heading_level = "H3"  # Default to H3 for pattern-based headings
                            
                            if heading_level:
                                pymupdf_headings.append({
                                    "level": heading_level,
                                    "text": text,
                                    "page": page_num + 1  # 1-indexed page numbers
                                })
                
                # Now try with PP-Structure for better layout analysis
                try:
                    # Convert PDF page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_np = np.array(img)
                    
                    # Preprocess the image
                    preprocessed_img = preprocess_image(img_np)
                    
                    # Process the image with PP-Structure
                    result = table_engine(preprocessed_img)
                    
                    # Extract title from the first page
                    if page_num == 0 and title == "Unknown Title":
                        try:
                            title = extract_title(result, pdf_path)
                        except Exception as e:
                            print(f"Warning: Error extracting title: {e}")
                    
                    # Detect document language from first page text
                    if page_num == 0:
                        try:
                            all_text = " ".join([item.get("text", "") for item in result if item.get("type") == "text"])
                            if all_text:
                                doc_language = detect_language(all_text)
                        except Exception as e:
                            print(f"Warning: Error detecting language: {e}")
                    
                    # Extract headings from the page using PP-Structure
                    try:
                        paddle_headings = classify_and_extract_headings(result, pdf_path, page_num, doc_language)
                        
                        # If we got headings from PP-Structure, use those instead of PyMuPDF headings
                        if paddle_headings:
                            outline.extend(paddle_headings)
                        else:
                            outline.extend(pymupdf_headings)
                    except Exception as e:
                        print(f"Warning: Error extracting headings from PP-Structure: {e}")
                        # Fallback to PyMuPDF headings
                        outline.extend(pymupdf_headings)
                
                except Exception as e:
                    print(f"Warning: PP-Structure processing failed: {e}")
                    # Fallback to PyMuPDF headings
                    outline.extend(pymupdf_headings)
            
            except Exception as e:
                print(f"Warning: Error processing page {page_num}: {e}")
                continue
        
        doc.close()
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        
        # Add at least one outline entry even if processing failed
        if not outline:
            outline.append({
                "level": "H1",
                "text": f"Document: {title}",
                "page": 1
            })
        
        return {"title": title, "outline": outline}
    
    # Deduplicate outline entries
    seen_entries = set()
    unique_outline = []
    for item in outline:
        entry_key = f"{item['level']}:{item['text']}:{item['page']}"
        if entry_key not in seen_entries:
            seen_entries.add(entry_key)
            unique_outline.append(item)
    
    # Sort outline by page number and then by position on page
    unique_outline.sort(key=lambda x: x["page"])
    
    # If no outline was found, create a basic one
    if not outline:
        print("No outline entries found, creating basic outline...")
        outline.append({
            "level": "H1",
            "text": title,
            "page": 1
        })
    
    # Post-process the outline to ensure hierarchical consistency
    processed_outline = []
    current_levels = {"H1": None, "H2": None, "H3": None}
    
    for item in unique_outline:
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
    print(f"Final title: {title}")
    print(f"Final outline entries: {len(processed_outline)}")
    
    return {"title": title, "outline": processed_outline}

def extract_toc_from_pdf(pdf_path):
    """Extract table of contents directly from PDF if available"""
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        total_pages = len(doc)  # Get total number of pages
        doc.close()
        
        if toc:
            outline = []
            for level, title, page in toc:
                if level <= 3:  # Only include up to H3
                    heading_level = f"H{level}"
                    outline.append({
                        "level": heading_level,
                        "text": title,
                        "page": page
                    })
            return outline, total_pages
    except Exception as e:
        print(f"Warning: Failed to extract TOC: {e}")
    
    return None, 0

def force_process_all_pages(pdf_path):
    """
    Force processing of all pages in the PDF to extract headings.
    This is a more aggressive approach when normal extraction fails.
    """
    outline = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"Force processing all {total_pages} pages in {pdf_path}")
        
        # Process each page individually
        for page_num in range(total_pages):
            try:
                print(f"Force processing page {page_num+1}/{total_pages}")
                page = doc[page_num]
                
                # Extract all text blocks with formatting
                blocks = page.get_text("dict").get("blocks", [])
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                        
                    for line_idx, line in enumerate(block["lines"]):
                        if "spans" not in line:
                            continue
                            
                        # Get text and font info
                        spans = line["spans"]
                        if not spans:
                            continue
                            
                        text = " ".join([span.get("text", "").strip() for span in spans])
                        if not text or len(text) < 3 or len(text) > 100:  # Skip empty or too long
                            continue
                        
                        # Get maximum font size and check if bold
                        max_font_size = 0
                        is_bold = False
                        for span in spans:
                            font_size = span.get("size", 0)
                            if font_size > max_font_size:
                                max_font_size = font_size
                            if "bold" in span.get("font", "").lower():
                                is_bold = True
                        
                        # Determine if this is likely a heading
                        is_heading = (
                            # Formatting-based criteria
                            (max_font_size > 11) or
                            (is_bold) or
                            # First line in a block
                            (line_idx == 0 and len(text) < 50) or
                            # Pattern-based criteria
                            (re.match(r'^\d+\.', text)) or
                            (re.match(r'^[A-Z][A-Za-z\s]+$', text) and len(text) < 40) or
                            (text.isupper() and len(text) < 30)
                        )
                        
                        if is_heading:
                            # Determine heading level
                            if max_font_size > 14 or (is_bold and max_font_size > 12):
                                level = "H1"
                            elif max_font_size > 12 or is_bold:
                                level = "H2"
                            else:
                                level = "H3"
                            
                            # Add to outline
                            outline.append({
                                "level": level,
                                "text": text,
                                "page": page_num + 1  # 1-indexed page numbers
                            })
                            print(f"Found heading on page {page_num+1}: {level} - {text}")
            
            except Exception as e:
                print(f"Warning: Error in force processing page {page_num+1}: {e}")
                continue
        
        doc.close()
        
    except Exception as e:
        print(f"Error in force_process_all_pages: {e}")
    
    return outline

def main():
    """Main function to process all PDFs in the input directory."""
    print(f"Starting PDF outline extraction with multi-language support")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = []
    for entry in os.scandir(INPUT_DIR):
        if entry.is_file() and entry.name.lower().endswith('.pdf') and not entry.name.endswith('.pdfZone.Identifier'):
            pdf_files.append(entry.path)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        try:
            # Get the base filename
            base_name = os.path.basename(pdf_path)
            output_name = os.path.splitext(base_name)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            
            # Get the total number of pages
            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()
                print(f"PDF {base_name} has {total_pages} pages")
            except Exception as e:
                print(f"Warning: Could not determine page count: {e}")
                total_pages = 0
            
            # First try to extract from embedded table of contents
            toc_outline, _ = extract_toc_from_pdf(pdf_path)
            
            if toc_outline and len(toc_outline) > 0:
                # Use the built-in TOC
                # Get title separately
                doc = fitz.open(pdf_path)
                title = doc.metadata.get("title", "")
                if not title or len(title.strip()) < 3:
                    # Extract first line of text as title
                    first_page_text = doc[0].get_text()
                    lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
                    if lines:
                        for line in lines[:5]:  # Check first 5 lines
                            if 3 < len(line) < 100:
                                title = line
                                break
                        if not title:
                            title = lines[0]
                doc.close()
                
                result = {
                    "title": title,
                    "outline": toc_outline
                }
            else:
                # Process the PDF with our extraction algorithm
                result = process_pdf(pdf_path)
                
                # Try forced processing for all pages to ensure we get headings from every page
                try:
                    doc = fitz.open(pdf_path)
                    total_pages = len(doc)
                    doc.close()
                    
                    # Force process if we have fewer headings than pages
                    if len(result["outline"]) < total_pages:
                        print(f"Found only {len(result['outline'])} headings for {total_pages} pages, trying forced processing")
                        force_outline = force_process_all_pages(pdf_path)
                        
                        # If we found more headings with forced processing, use those instead
                        if len(force_outline) > len(result["outline"]):
                            print(f"Force processing found {len(force_outline)} headings, using these instead")
                            result["outline"] = force_outline
                except Exception as e:
                    print(f"Warning: Could not check page count or force process: {e}")
                    
                # Ensure we have at least a default title
            if not result["title"] or result["title"] == "Unknown Title":
                # Try to use filename as title
                filename_title = os.path.splitext(base_name)[0]
                if len(filename_title) > 3:
                    result["title"] = filename_title
            
            # If we have very few or no headings, try a more aggressive extraction
            if not result["outline"] or (total_pages > 3 and len(result["outline"]) < 3):
                try:
                    doc = fitz.open(pdf_path)
                    fallback_outline = []
                    
                    # Scan EVERY page for potential headings
                    for page_num in range(len(doc)):
                        print(f"Processing page {page_num+1}/{len(doc)} of {base_name}")
                        page = doc[page_num]
                        
                        # Get text with detailed formatting info
                        blocks = page.get_text("dict").get("blocks", [])
                        
                        for block in blocks:
                            if "lines" not in block:
                                continue
                            
                            for line_idx, line in enumerate(block["lines"]):
                                if "spans" not in line:
                                    continue
                                    
                                # Get text from spans
                                text = " ".join([span.get("text", "").strip() for span in line["spans"]])
                                if not text or len(text) < 3 or len(text) > 100:
                                    continue
                                
                                # Get formatting attributes
                                is_bold = False
                                font_size = 0
                                if line["spans"]:
                                    span = line["spans"][0]
                                    font_size = span.get("size", 0)
                                    is_bold = "bold" in span.get("font", "").lower()
                                
                                # Improved heading detection: 
                                # - Short text (< 80 chars)
                                # - Large font or bold text
                                # - Not ending with punctuation 
                                # - Or has numbering pattern
                                is_potential_heading = (
                                    len(text) < 80 and
                                    (font_size > 10 or is_bold or
                                     line_idx == 0 and len(text) < 40) and
                                    (not text[-1] in ".,:;?!" or 
                                     re.match(r'^\d+\.', text) or 
                                     re.match(r'^[A-Z][A-Za-z\s]+$', text) or
                                     re.match(r'^[IVXLCDM]+\.\s', text))
                                )
                                
                                if is_potential_heading:
                                    # Determine level based on formatting and position
                                    if font_size > 14 or (is_bold and font_size > 12) or page_num == 0 and line_idx == 0:
                                        level = "H1"
                                    elif font_size > 12 or is_bold:
                                        level = "H2"
                                    else:
                                        level = "H3"
                                        
                                    fallback_outline.append({
                                        "level": level,
                                        "text": text,
                                        "page": page_num + 1
                                    })
                    
                    doc.close()
                    
                    # Add fallback outline if we found potential headings
                    if fallback_outline:
                        # Only replace if we found more headings or had none before
                        if len(fallback_outline) > len(result["outline"]):
                            result["outline"] = fallback_outline
                        
                except Exception as e:
                    print(f"Warning: Fallback extraction failed: {e}")
            
            # Write the result to a JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Output written to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            # Ensure we always create an output file
            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(pdf_path))[0] + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"title": os.path.basename(pdf_path), "outline": []}, f, indent=2)
    
    print("PDF outline extraction complete")

if __name__ == "__main__":
    main() 