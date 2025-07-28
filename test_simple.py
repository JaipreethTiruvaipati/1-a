#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import time
import langdetect

# Constants
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

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

def process_pdf_simple(pdf_path, lang=None):
    """
    Process a single PDF file and extract its outline using only PyMuPDF.
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
                # Extract text with detailed formatting info
                page = doc[page_num]
                page_text = page.get_text("dict")
                
                # Extract language if not detected yet
                if page_num == 0:
                    try:
                        text_content = page.get_text()
                        if text_content:
                            doc_language = detect_language(text_content)
                    except:
                        pass
                
                # Analyze blocks directly from PyMuPDF for headings
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
                                outline.append({
                                    "level": heading_level,
                                    "text": text,
                                    "page": page_num + 1  # 1-indexed page numbers
                                })
                                print(f"Found heading on page {page_num+1}: {heading_level} - {text}")
            
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
    
    print(f"Processed in {time.time() - start_time:.2f} seconds")
    print(f"Final title: {title}")
    print(f"Final outline entries: {len(unique_outline)}")
    print(f"Detected language: {doc_language}")
    
    return {"title": title, "outline": unique_outline, "language": doc_language}

def main():
    """Main function to process all PDFs in the input directory."""
    print(f"Starting PDF outline extraction with multi-language support (Simple Version)")
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
            
            # Process the PDF
            result = process_pdf_simple(pdf_path)
            
            # Write the result to a JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Output written to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            # Ensure we always create an output file
            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(pdf_path))[0] + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"title": os.path.basename(pdf_path), "outline": [], "language": "en"}, f, indent=2)
    
    print("PDF outline extraction complete")

if __name__ == "__main__":
    main() 