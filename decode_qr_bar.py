
from pyzbar.pyzbar import decode
import os
import cv2 

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")

# Initialize readers
pyzbar_reader = decode

image_files = (
    'card_national.jpg', 'card2.png', 'qr.png', 'qr2.png', 
    'qr3.png', 'qr4.png', 'voters_blur.png', 'voters1.jpeg', 
    'voters2.jpg', 'votters_id.jpg',
    'amalilaptop.jpg', 'basitv.jpg', 'basitv2.jpg',
    'bar.png', 'bar2.png'
)

for img_name in image_files:
    img_path = os.path.join(assets_dir, img_name)
    
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue

    # Decode with all three readers
    pyzbar_out = pyzbar_reader(image=img)

    pyzbar_out = tuple(out.data.decode('utf-8') for out in pyzbar_out)

    print(f"Image: {img_name}")
    print(f"  pyzbar:  {pyzbar_out}")
    print()
