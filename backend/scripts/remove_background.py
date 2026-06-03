import cv2
import numpy as np
import os

def remove_background(img_path, output_path, lo_tolerance=45, up_tolerance=45):
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return False
        
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Start with a mask initialized to all 0s (will contain floodfill area)
    # The mask for floodFill must be 2 pixels wider and taller than the image
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # We flood fill from the four corners.
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for pt in corners:
        cv2.floodFill(img, mask, pt, (0, 0, 0),
                      loDiff=(lo_tolerance, lo_tolerance, lo_tolerance),
                      upDiff=(up_tolerance, up_tolerance, up_tolerance),
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)
                      
    # Also fill from border edges to catch any isolated white blocks
    for x in range(0, w, 20):
        cv2.floodFill(img, mask, (x, 0), (0, 0, 0),
                      loDiff=(lo_tolerance, lo_tolerance, lo_tolerance),
                      upDiff=(up_tolerance, up_tolerance, up_tolerance),
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(img, mask, (x, h - 1), (0, 0, 0),
                      loDiff=(lo_tolerance, lo_tolerance, lo_tolerance),
                      upDiff=(up_tolerance, up_tolerance, up_tolerance),
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)
    for y in range(0, h, 20):
        cv2.floodFill(img, mask, (0, y), (0, 0, 0),
                      loDiff=(lo_tolerance, lo_tolerance, lo_tolerance),
                      upDiff=(up_tolerance, up_tolerance, up_tolerance),
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(img, mask, (w - 1, y), (0, 0, 0),
                      loDiff=(lo_tolerance, lo_tolerance, lo_tolerance),
                      upDiff=(up_tolerance, up_tolerance, up_tolerance),
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)

    # The filled region in mask is marked with 1. We invert it to get our alpha channel.
    filled_area = mask[1:-1, 1:-1]
    alpha = np.ones((h, w), dtype=np.uint8) * 255
    alpha[filled_area > 0] = 0
    
    # Apply morphology opening to clean up edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    
    # Smooth alpha borders
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    
    # Merge back to BGRA
    bgra = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha])
    cv2.imwrite(output_path, bgra)
    print(f"Processed background removal for {img_path} -> {output_path}")
    return True

if __name__ == "__main__":
    assets_dir = "/Users/belisozcelik/Facial-Image-Warping/backend/assets/clothing"
    
    # Process shirt_model (checkered background, high tolerance needed)
    remove_background(
        os.path.join(assets_dir, "shirt_model.png"),
        os.path.join(assets_dir, "shirt_model.png"),
        lo_tolerance=55,
        up_tolerance=55
    )
    
    # Process hoodie_model (white background, standard tolerance is fine)
    remove_background(
        os.path.join(assets_dir, "hoodie_model.png"),
        os.path.join(assets_dir, "hoodie_model.png"),
        lo_tolerance=35,
        up_tolerance=35
    )
