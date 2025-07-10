#!/usr/bin/env python3
"""
Binary Log File Analyzer

This script analyzes binary log files to understand their structure
and determine the best approach for parsing them.
"""

import os
import struct
import binascii


def analyze_binary_file(filename):
    """Analyze a binary file to understand its structure"""
    print(f"Analyzing binary file: {filename}")
    print("=" * 50)
    
    try:
        with open(filename, 'rb') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Back to beginning
            
            print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            # Read first chunk
            chunk_size = min(1024, file_size)
            data = f.read(chunk_size)
            
            print(f"\nFirst {len(data)} bytes:")
            print("-" * 30)
            
            # Try to decode as text
            try:
                text_data = data.decode('utf-8')
                print("✅ File appears to be UTF-8 text:")
                print(text_data[:500])  # First 500 chars
                return "text"
            except UnicodeDecodeError:
                pass
            
            # Try other encodings
            for encoding in ['ascii', 'latin-1', 'cp1252']:
                try:
                    text_data = data.decode(encoding)
                    print(f"✅ File appears to be {encoding} text:")
                    print(text_data[:500])  # First 500 chars
                    return "text"
                except UnicodeDecodeError:
                    continue
            
            # If not text, analyze as binary
            print("🔍 File appears to be binary data")
            print("\nHex dump (first 200 bytes):")
            hex_data = binascii.hexlify(data[:200]).decode('ascii')
            for i in range(0, len(hex_data), 32):
                print(f"{i//2:04x}: {hex_data[i:i+32]}")
            
            # Look for common patterns
            print("\n🔍 Pattern Analysis:")
            
            # Check for repeated structures
            print(f"First 16 bytes: {data[:16].hex()}")
            print(f"Bytes 16-32: {data[16:32].hex()}")
            print(f"Bytes 32-48: {data[32:48].hex()}")
            
            # Look for common binary log signatures
            if data[:4] == b'RIFF':
                print("📁 Might be RIFF format")
            elif data[:4] == b'\x50\x4B\x03\x04':
                print("📁 Might be ZIP format")
            elif data[:2] == b'BM':
                print("📁 Might be BMP format")
            elif data[:4] == b'\x89PNG':
                print("📁 Might be PNG format")
            elif data[:3] == b'ID3':
                print("📁 Might be MP3 format")
            
            # Check for timestamp patterns (Unix timestamps)
            for i in range(0, len(data)-4, 4):
                timestamp = struct.unpack('<I', data[i:i+4])[0]
                if 1000000000 < timestamp < 2000000000:  # Reasonable timestamp range
                    from datetime import datetime
                    dt = datetime.fromtimestamp(timestamp)
                    print(f"📅 Possible timestamp at offset {i}: {dt}")
                    break
            
            # Check for string patterns
            strings = []
            current_string = ""
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) > 3:
                        strings.append(current_string)
                    current_string = ""
            
            if strings:
                print(f"\n📝 Found text strings: {strings[:10]}")  # First 10 strings
            
            return "binary"
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return "error"


def suggest_parsing_approach(file_type):
    """Suggest parsing approach based on file analysis"""
    print("\n" + "=" * 50)
    print("🎯 PARSING RECOMMENDATIONS")
    print("=" * 50)
    
    if file_type == "text":
        print("✅ File is text-based - can use existing LogNarrator parser!")
        print("💡 Try: python main.py -f \"Main_1 (2)\"")
        
    elif file_type == "binary":
        print("🔧 File is binary - needs special handling")
        print("💡 Options:")
        print("  1. Find documentation for the binary format")
        print("  2. Use a hex editor to analyze structure")
        print("  3. Check if vendor provides conversion tools")
        print("  4. Create custom binary parser")
        
    else:
        print("❌ Unable to determine file type")


if __name__ == "__main__":
    filename = "Main_1 (2)"
    if os.path.exists(filename):
        file_type = analyze_binary_file(filename)
        suggest_parsing_approach(file_type)
    else:
        print(f"❌ File not found: {filename}") 