#!/usr/bin/env python
"""
f1_on_road_fullscreen.py

- RD.jpg   : road background
- car1.jpg : first F1 car on white background
- car2.jpg : second F1 car on white background

Fullscreen animation:
Both cars drive along the road left→right in a loop,
car2 is faster than car1.

Quit with 'q' or Esc.

Requirements (Anaconda):
    conda install -c conda-forge opencv
"""

import cv2
import numpy as np

# --------------------------------------------------
# 1. Load images
# --------------------------------------------------
bg_path   = "Downloads/RD.jpg"
car1_path = "Downloads/car1.jpg"
car2_path = "Downloads/car3.jpg"

bg   = cv2.imread(bg_path, cv2.IMREAD_COLOR)
car1 = cv2.imread(car1_path, cv2.IMREAD_COLOR)
car2 = cv2.imread(car2_path, cv2.IMREAD_COLOR)

if bg is None:
    raise FileNotFoundError(f"Could not read {bg_path}")
if car1 is None:
    raise FileNotFoundError(f"Could not read {car1_path}")
if car2 is None:
    raise FileNotFoundError(f"Could not read {car2_path}")

# --------------------------------------------------
# 2. Resize background to screen size (fullscreen)
# --------------------------------------------------
try:
    import tkinter as tk
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    bg = cv2.resize(bg, (screen_w, screen_h), interpolation=cv2.INTER_AREA)
    print(f"Background resized to screen: {screen_w}x{screen_h}")
except Exception as e:
    print("Could not get screen size via tkinter, using original size:", e)

h_bg, w_bg = bg.shape[:2]

# --------------------------------------------------
# 3. Helper to create mask (remove white background)
# --------------------------------------------------
def make_fg_mask(img, thresh=240):
    """
    Create a mask for non-white pixels.
    Assumes object on (near) white background.
    """
    mask = np.any(img < thresh, axis=2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask

# --------------------------------------------------
# 4. Prepare cars: flip so they face right, remove bg, resize
# --------------------------------------------------

# Flip horizontally so cars point to the right (direction of motion)
car1 = cv2.flip(car1, 1)
#car2 = cv2.flip(car2, 1)

mask1 = make_fg_mask(car1, thresh=240)
mask2 = make_fg_mask(car2, thresh=240)

h1, w1 = car1.shape[:2]
h2, w2 = car2.shape[:2]

# Make both cars the SAME height
desired_h = int(h_bg * 0.25)  # 25% of screen height

scale1 = desired_h / float(h1)
scale2 = desired_h / float(h2)

new_w1, new_h1 = int(w1 * scale1), desired_h
new_w2, new_h2 = int(w2 * scale2), desired_h

car1 = cv2.resize(car1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)
mask1 = cv2.resize(mask1, (new_w1, new_h1), interpolation=cv2.INTER_AREA)

car2 = cv2.resize(car2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
mask2 = cv2.resize(mask2, (new_w2, new_h2), interpolation=cv2.INTER_AREA)

h1, w1 = car1.shape[:2]
h2, w2 = car2.shape[:2]

# --------------------------------------------------
# 5. Overlay helper
# --------------------------------------------------
def overlay_with_mask(bg_img, fg_img, mask, x, y):
    """
    Overlay fg_img onto bg_img at (x, y) using single-channel mask.
    Returns a new image, bg_img is not modified in-place.
    """
    bg_h, bg_w = bg_img.shape[:2]
    fg_h, fg_w = fg_img.shape[:2]

    # Clip to background
    if x < 0:
        fg_x1 = -x
        bg_x1 = 0
    else:
        fg_x1 = 0
        bg_x1 = x

    if y < 0:
        fg_y1 = -y
        bg_y1 = 0
    else:
        fg_y1 = 0
        bg_y1 = y

    bg_x2 = min(bg_x1 + fg_w - fg_x1, bg_w)
    bg_y2 = min(bg_y1 + fg_h - fg_y1, bg_h)

    fg_x2 = fg_x1 + (bg_x2 - bg_x1)
    fg_y2 = fg_y1 + (bg_y2 - bg_y1)

    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return bg_img.copy()

    roi_bg   = bg_img[bg_y1:bg_y2, bg_x1:bg_x2]
    roi_fg   = fg_img[fg_y1:fg_y2, fg_x1:fg_x2]
    roi_mask = mask[fg_y1:fg_y2, fg_x1:fg_x2]

    alpha = (roi_mask.astype(np.float32) / 255.0)[:, :, None]

    blended = roi_bg.astype(np.float32) * (1.0 - alpha) + \
              roi_fg.astype(np.float32) * alpha

    out = bg_img.copy()
    out[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)
    return out

# --------------------------------------------------
# 6. Animation setup
# --------------------------------------------------
# Place both cars near the road
margin_from_bottom = int(h_bg * 0.08)

y1 = h_bg - h1 - margin_from_bottom
y2 = h_bg - h2 - int(margin_from_bottom * 3)  # a bit higher, like another lane

y1 = max(0, y1)
y2 = max(0, y2)

# Start positions (off-screen to the left)
x1 = -w1 * 1.2     # slower car
x2 = -w2 * 2.2     # faster car, starts further back

# Speeds: car2 faster than car1
speed1 = max(6, w_bg // 160)   # slower
speed2 = max(11, w_bg // 95)   # faster

window_name = "F1 Road Race (press 'q' or Esc to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --------------------------------------------------
# 7. Main loop
# --------------------------------------------------
while True:
    frame = bg.copy()

    frame = overlay_with_mask(frame, car1, mask1, int(x1), y1)
    frame = overlay_with_mask(frame, car2, mask2, int(x2), y2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(25) & 0xFF  # ~40 fps
    if key == ord('q') or key == 27:  # Esc
        break

    # Update positions (moving left -> right)
    x1 += speed1
    x2 += speed2

    # Loop each car back to the left when it exits on the right
    if x1 > w_bg:
        x1 = -w1 * 1.2
    if x2 > w_bg:
        x2 = -w2 * 2.2

cv2.destroyAllWindows()
