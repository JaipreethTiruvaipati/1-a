#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for PDF outline extraction.
This script can be used to test the solution without Docker.
"""

import os
import sys
import json
from main import process_pdf

def test_pdf_extraction(pdf_path, output_path):
    """Test the PDF extraction on a single file."""
    print(f"Testing PDF extraction on: {pdf_path}")
    
    # Process the PDF
    result = process_pdf(pdf_path)
    
    # Print the result
    print("\nExtracted Title:")
    print(result["title"])
    
    print("\nExtracted Outline:")
    for item in result["outline"]:
        print(f"{item['level']} - {item['text']} (Page {item['page']})")
    
    # Write the result to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nOutput written to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_script.py <path_to_pdf> [output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = os.path.splitext(pdf_path)[0] + ".json"
    
    test_pdf_extraction(pdf_path, output_path) 