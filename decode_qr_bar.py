from pyzbar.pyzbar import decode
import os
import cv2 
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(script_dir, "assets")

image_files = 'schid.jpg'
img_path = os.path.join(assets_dir, image_files)

if not os.path.exists(img_path):
    print(f"File not found: {img_path}")

img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load: {img_path}")

# Decode barcode
pyzbar_results = decode(img)

# Format output similar to MRZScanner
result = {}
if pyzbar_results:
    polygon_points = []
    barcode_texts = []
    
    for res in pyzbar_results:
        # Extract text
        text = res.data.decode('utf-8') if res.data else ""
        barcode_texts.append(text)
        
        # Extract polygon if available
        if hasattr(res, 'polygon') and res.polygon:
            polygon = np.array([[p.x, p.y] for p in res.polygon], dtype=np.float32)
            if polygon.size > 0:
                polygon_points = polygon
    
    result = {
        'barcode_polygon': polygon_points.tolist() if polygon_points.size > 0 else None,
        'barcode_texts': barcode_texts,
        'msg': f"Found {len(pyzbar_results)} barcode(s)"
    }
else:
    result = {
        'barcode_polygon': None,
        'barcode_texts': [],
        'msg': 'No barcode detected'
    }

# Create final output dictionary (similar to MRZScanner format)
output = {
    "success": len(pyzbar_results) > 0,
    "barcode_polygon": result['barcode_polygon'],
    "barcode_texts": result['barcode_texts'],
    "msg": result['msg']
}

print(f"Image: {image_files}")
print(f"Output: {output}")