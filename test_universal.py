#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import langdetect
from collections import Counter

def detect_language_universal(text):
    """
    Universal language detection that works with any text content.
    Uses multiple strategies to detect language reliably.
    """
    if not text or len(text.strip()) < 5:
        return 'en'  # Default to English if text is too short
    
    # Strategy 1: Character set detection (most reliable for non-Latin scripts)
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
    
    # Strategy 2: langdetect with multiple samples
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
    
    print(f"Language detection failed, defaulting to English")
    return 'en'  # Default to English if all detection methods fail

def test_language_detection():
    """Test the universal language detection with sample texts."""
    
    test_texts = [
        # English
        ("This is a sample English text for testing language detection.", "en"),
        
        # Japanese
        ("これは日本語のサンプルテキストです。言語検出のテスト用です。", "ja"),
        
        # Chinese
        ("这是一个中文样本文本。用于语言检测测试。", "zh"),
        
        # Korean
        ("이것은 한국어 샘플 텍스트입니다. 언어 감지 테스트용입니다.", "ko"),
        
        # Arabic
        ("هذا نص عربي للاختبار. يستخدم لاختبار اكتشاف اللغة.", "ar"),
        
        # Thai
        ("นี่คือตัวอย่างข้อความภาษาไทย สำหรับทดสอบการตรวจจับภาษา", "th"),
        
        # Hindi
        ("यह हिंदी का नमूना पाठ है। भाषा पहचान के लिए परीक्षण।", "hi"),
        
        # Russian
        ("Это образец русского текста. Для тестирования определения языка.", "ru"),
        
        # Tamil
        ("இது தமிழ் மாதிரி உரை. மொழி கண்டறிதல் சோதனைக்கு.", "ta"),
        
        # Kannada
        ("ಇದು ಕನ್ನಡ ಮಾದರಿ ಪಠ್ಯ. ಭಾಷೆ ಪತ್ತೆಹಚ್ಚುವ ಪರೀಕ್ಷೆಗಾಗಿ.", "kn"),
        
        # Malayalam
        ("ഇത് മലയാളം സാമ്പിൾ ടെക്സ്റ്റാണ്. ഭാഷ കണ്ടെത്തുന്നതിനുള്ള പരീക്ഷണം.", "ml"),
        
        # Sinhala
        ("මෙය සිංහල සාම්පල් පාඨයකි. භාෂා හඳුනාගැනීමේ පරීක්ෂණය සඳහා.", "si"),
        
        # Lao
        ("ນີ້ແມ່ນຕົວຢ່າງຂໍ້ຄວາມພາສາລາວ. ສຳລັບການທົດສອບການກວດສອບພາສາ", "lo"),
        
        # Tibetan
        ("འདི་ནི་བོད་ཡིག་གི་དཔེར་ན་ཡིག་ཆ་ཞིག་རེད། ཡིག་སྐད་ཤེས་པའི་ཆ་རྐྱེན་ཆེད།", "bo"),
        
        # Myanmar
        ("ဤသည်မှာ မြန်မာဘာသာ နမူနာ စာသားဖြစ်သည်။ ဘာသာစကား ရှာဖွေရန် စမ်းသပ်မှုအတွက်။", "my"),
        
        # Ethiopic (Amharic)
        ("ይህ አማርኛ ናሙና ጽሑፍ ነው። ቋንቋ ለማወቅ ለሚደረገው ሙከራ።", "am"),
        
        # Georgian
        ("ეს არის ქართული ნიმუშის ტექსტი. ენის აღმოსაჩენად ტესტირებისთვის.", "ka"),
        
        # Greek
        ("Αυτό είναι ένα δείγμα ελληνικού κειμένου. Για δοκιμή ανίχνευσης γλώσσας.", "el"),
        
        # Mixed text (English + Chinese)
        ("This is a mixed text. 这是混合文本。", "zh"),  # Should detect Chinese due to characters
    ]
    
    print("Testing Universal Language Detection")
    print("=" * 50)
    
    correct = 0
    total = len(test_texts)
    
    for i, (text, expected) in enumerate(test_texts, 1):
        detected = detect_language_universal(text)
        status = "✓" if detected == expected else "✗"
        print(f"{i:2d}. {status} Expected: {expected:2s}, Detected: {detected:2s}")
        print(f"    Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print()
        
        if detected == expected:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy

if __name__ == "__main__":
    test_language_detection() 