import cv2
import numpy as np
import random
import os
import re
from tqdm import tqdm
from rembg import remove, new_session

# ---------------- CONFIG ---------------- #
INPUT_DIR = r"C:\Users\KIIT0001\Downloads\archive (1)\images_compressed"
STAIN_LIB_DIR =  r"C:\Users\KIIT0001\Pictures\stain marks" 
OUTPUT_ROOT = r"C:\Users\KIIT0001\Downloads\archive (1)\stained_2"
session = new_session()

# Create levels 0 through 5
for i in range(6):
    os.makedirs(os.path.join(OUTPUT_ROOT, f"Level_{i}", "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, f"Level_{i}", "masks"), exist_ok=True)

# ---------------- CORE FUNCTIONS ---------------- #

def get_stain_by_level(target_level):
    """Scans library for 'level_X.Y' naming convention."""
    all_stains = []
    for root, dirs, files in os.walk(STAIN_LIB_DIR):
        for file in files:
            if re.search(rf"level_{target_level}\.", file):
                all_stains.append(os.path.join(root, file))
    return random.choice(all_stains) if all_stains else None

def blend_stain(image, stain_img, garment_mask):
    """Performs Multiply blend restricted to garment area with boundary clipping."""
    h, w = image.shape[:2]
    
    # 1. Random Resize (15% to 40% of garment width)
    scale = random.uniform(0.15, 0.4)
    nw = int(w * scale)
    nh = int(stain_img.shape[0] * (nw / stain_img.shape[1]))
    
    # Ensure nh and nw are at least 1 pixel
    nw, nh = max(1, nw), max(1, nh)
    stain_resized = cv2.resize(stain_img, (nw, nh))

    # 2. Random Placement
    x = random.randint(0, max(0, w - nw))
    y = random.randint(0, max(0, h - nh))

    # 3. BOUNDARY CLIP: Calculate actual slice dimensions to prevent BroadCast errors
    # This ensures the stain never exceeds the background image boundaries
    actual_h = min(nh, h - y)
    actual_w = min(nw, w - x)
    stain_resized = stain_resized[:actual_h, :actual_w]

    local_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 4. Extract Mask from Alpha or Grayscale
    if stain_resized.shape[2] == 4: # Handle PNG Alpha
        local_mask[y:y+actual_h, x:x+actual_w] = stain_resized[:, :, 3]
    else: # Fallback for non-alpha images
        gray = cv2.cvtColor(stain_resized, cv2.COLOR_BGR2GRAY)
        local_mask[y:y+actual_h, x:x+actual_w] = 255 - gray 

    # 5. Constrain and Blur Mask
    final_mask = cv2.bitwise_and(local_mask, garment_mask)
    alpha = cv2.GaussianBlur(final_mask, (11, 11), 0).astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    # 6. Apply Multiply Blend
    stain_layer = np.full_like(image, 255)
    stain_layer[y:y+actual_h, x:x+actual_w] = stain_resized[:, :, :3]
    
    multiplied = (image.astype(float) * (stain_layer.astype(float) / 255.0)).astype(np.uint8)
    output = (image * (1 - alpha) + multiplied * alpha).astype(np.uint8)
    
    return output, final_mask

def apply_stain_logic(base_img, g_mask, target_level):
    """Defines how many and which stain files to combine for each level."""
    combined_mask = np.zeros(base_img.shape[:2], dtype=np.uint8)
    working_img = base_img.copy()

    strategy = {
        1: [1],       
        2: [2],       
        3: [3],       
        4: [2, 2, 1], 
        5: [3, 2, 2]  
    }

    levels_to_pull = strategy.get(target_level, [])
    for s_level in levels_to_pull:
        path = get_stain_by_level(s_level)
        if path:
            stain_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if stain_img is not None:
                working_img, m = blend_stain(working_img, stain_img, g_mask)
                combined_mask = cv2.bitwise_or(combined_mask, m)

    return working_img, combined_mask

# ---------------- EXECUTION ---------------- #

image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in tqdm(image_files):
    img = cv2.imread(os.path.join(INPUT_DIR, img_name))
    if img is None: continue

    # Detect Garment boundary
    g_mask = np.array(remove(img, session=session, only_mask=True))

    # Randomly pick a level for this image
    target_level = random.randint(0, 5)
    
    if target_level == 0:
        final_img, final_mask = img, np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        final_img, final_mask = apply_stain_logic(img, g_mask, target_level)

    # Save to level-specific folders
    level_folder = f"Level_{target_level}"
    out_img_path = os.path.join(OUTPUT_ROOT, level_folder, "images", img_name)
    out_mask_path = os.path.join(OUTPUT_ROOT, level_folder, "masks", img_name)

    cv2.imwrite(out_img_path, final_img)
    cv2.imwrite(out_mask_path, final_mask)

print(f"\n✅ Processing complete. Find your levels in: {OUTPUT_ROOT}")