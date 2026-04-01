"""
generate_plates_simple.py
License Plate Generator - Simple version without blur
"""

import os
import sys

# Tắt warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import warnings

warnings.filterwarnings('ignore')
from tensorflow.keras.datasets import mnist


class LicensePlateGenerator:
    def __init__(self, output_dir='data/plate', verbose=True):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print("Loading MNIST dataset...")
        
        (self.mnist_train, self.mnist_labels_train), (_, _) = mnist.load_data()
        
        if self.verbose:
            print("Creating synthetic letters...")
        
        self.letters_images, self.letters_labels = self.create_synthetic_letters()
        
        if self.verbose:
            print(f"Ready: {len(self.mnist_train)} digits, {len(self.letters_images)} letters")
    
    def create_synthetic_letters(self):
        """Tạo chữ cái synthetic (không có blur)"""
        letters = []
        labels = []
        
        for letter_idx in range(26):
            letter_char = chr(65 + letter_idx)
            
            if self.verbose and letter_idx % 5 == 0:
                print(f"  Generating {letter_char}...")
            
            for _ in range(200):
                img = Image.new('L', (28, 28), color=255)
                draw = ImageDraw.Draw(img)
                
                try:
                    if os.name == 'nt':
                        font = ImageFont.truetype("arial.ttf", 20)
                    else:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), letter_char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (28 - text_width) // 2
                y = (28 - text_height) // 2
                
                draw.text((x, y), letter_char, fill=0, font=font)
                
                img_array = np.array(img, dtype=np.uint8)
                
                # Chỉ thêm nhiễu
                noise = np.random.randint(0, 20, img_array.shape, dtype=np.uint8)
                img_array = np.clip(img_array.astype(np.int16) - noise, 0, 255).astype(np.uint8)
                
                letters.append(img_array)
                labels.append(letter_idx)
        
        return np.array(letters), np.array(labels)
    
    def get_digit_image(self, digit_char):
        """Lấy ảnh chữ số phù hợp"""
        matching = []
        for idx, label in enumerate(self.mnist_labels_train):
            if str(label) == digit_char:
                matching.append(self.mnist_train[idx])
                if len(matching) > 20:
                    break
        
        if matching:
            return random.choice(matching)
        
        idx = random.randint(0, len(self.mnist_train) - 1)
        return self.mnist_train[idx]
    
    def get_letter_image(self, letter_char):
        """Lấy ảnh chữ cái phù hợp"""
        target_idx = ord(letter_char) - 65
        
        matching = []
        for idx, label in enumerate(self.letters_labels):
            if label == target_idx:
                matching.append(self.letters_images[idx])
                if len(matching) > 20:
                    break
        
        if matching:
            return random.choice(matching)
        
        # Fallback
        img = Image.new('L', (28, 28), color=255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((4, 4), letter_char, fill=0, font=font)
        return np.array(img, dtype=np.uint8)
    
    def create_license_plate(self, plate_number):
        """Tạo ảnh biển số"""
        char_size = 64
        padding = 15
        total_width = len(plate_number) * char_size + (len(plate_number) - 1) * padding + 60
        total_height = char_size + 60
        
        plate = Image.new('RGB', (total_width, total_height), color=(0, 105, 180))
        draw = ImageDraw.Draw(plate)
        draw.rectangle([(8, 8), (total_width - 8, total_height - 8)], 
                       outline=(255, 255, 255), width=4)
        
        x = 30
        y = 25
        
        for char in plate_number:
            if char.isdigit():
                img_array = self.get_digit_image(char)
            else:
                img_array = self.get_letter_image(char)
            
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            pil_img = Image.fromarray(img_array)
            pil_img = pil_img.resize((char_size, char_size), Image.Resampling.LANCZOS)
            mask = pil_img.point(lambda p: 255 if p < 128 else 0, '1')
            char_img = Image.new('RGB', (char_size, char_size), color=(255, 255, 255))
            plate.paste(char_img, (x, y), mask)
            
            x += char_size + padding
        
        plate_img = np.array(plate, dtype=np.uint8)
        noise = np.random.randint(0, 10, plate_img.shape, dtype=np.uint8)
        plate_img = np.clip(plate_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return plate_img
    
    def generate_random_plate(self):
        """Tạo biển số ngẫu nhiên"""
        part1 = f"{random.randint(10, 99):02d}"
        part2 = chr(65 + random.randint(0, 25))
        part3 = f"{random.randint(10000, 99999):05d}"
        return f"{part1}{part2}{part3}"
    
    def generate_batch(self, num_plates=100):
        """Tạo batch biển số"""
        results = []
        
        for i in range(num_plates):
            plate_number = self.generate_random_plate()
            plate_img = self.create_license_plate(plate_number)
            
            filename = f"plate_{i+1:04d}_{plate_number}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            plate_img_bgr = cv2.cvtColor(plate_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, plate_img_bgr)
            
            results.append((plate_number, filepath))
            
            if (i + 1) % 20 == 0:
                print(f"Generated: {i+1}/{num_plates}")
        
        return results


def main():
    print("="*50)
    print("License Plate Generator")
    print("="*50)
    
    generator = LicensePlateGenerator(output_dir='data/plate', verbose=True)
    
    print("\nGenerating 100 license plates...")
    plates = generator.generate_batch(100)
    
    print(f"\n✅ Completed! Generated {len(plates)} plates")
    print(f"📁 Saved to: data/plate/")
    
    print("\n📝 First 10 plates:")
    for i, (plate, path) in enumerate(plates[:10]):
        print(f"  {i+1}. {plate}")
    
    list_file = 'data/plate/plate_list.txt'
    with open(list_file, 'w') as f:
        for plate, _ in plates:
            f.write(f"{plate}\n")
    
    print(f"\n💾 Plate list saved to: {list_file}")
    print("\n⚠️  Note: Only images are saved, no labels provided.")


if __name__ == "__main__":
    main()