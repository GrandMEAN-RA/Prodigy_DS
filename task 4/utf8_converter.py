# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 16:12:21 2025

@author: EBUNOLUWASIMI
"""

import chardet

def convert_to_utf8(input_file, output_file=None):
    # Step 1: Detect the encoding
    with open(input_file, "rb") as f:
        raw_data = f.read(100000)  # read first 100KB for detection
        result = chardet.detect(raw_data)
        detected_encoding = result["encoding"]

    print(f"Detected encoding: {detected_encoding}")

    # Step 2: Read with detected encoding
    with open(input_file, "r", encoding=detected_encoding, errors="replace") as f:
        content = f.read()

    # Step 3: Save as UTF-8
    if output_file is None:
        output_file = input_file.rsplit(".", 1)[0] + "_utf8.csv"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"File saved as UTF-8: {output_file}")
    
    return output_file
