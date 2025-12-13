#!/usr/bin/env python
"""
motor_in_city_fullscreen.py

- Reads City.jpg (background) and Motor.jpg (motorcycle on white bg)
- Removes the white background from the motorcycle
- Resizes the city to full-screen resolution
- Animates the motorcycle moving continuously across the city
- Shows the animation in a fullscreen window (no video file saved)

Quit the animation window by pressing 'q' or Esc.

Requirements (Anaconda):
    conda install -c conda-forge opencv
"""

import cv2
import numpy as np

# ----------------------------
# 1. Load images
# ----------------------------
city_path = "City.jpg"
motor_path = "Motor.jpg"

city = cv2.imread(city_path, cv2.IMREAD_COLOR)
motor = cv2.imread(motor_path, cv2.IMREAD_COLOR)

if city is None:
    raise FileNotFoundError(f"Could not read {city_path}")
if motor is None:
    raise FileNotFoundError(f"Could not read {motor_path}")

# ----------------------------
# 2. Resize city to full screen
# ----------------------------
try:
    import tkinter as tk
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    # Resize background to exactly match screen size
    city = cv2.resize(city, (screen_w, screen_h), interpolation=cv2.INTER_AREA)
    print(f"Resized city to screen: {screen_w}x{screen_h}")
except Exception as e:
    # Fallback: use original size if tkinter not available
    print("Could not determine screen size, using original city size:", e)

h_bg, w_bg = city.shape[:2]

# ----------------------------
# 3. Remove background of motor
#    (assumes white / very light background)
# ----------------------------
threshold = 240  # tweak if needed (230–250)
fg_mask = np.any(motor < threshold, axis=2).astype(np.uint8) * 255

kernel = np.ones((5, 5), np.uint8)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

# ----------------------------
# 4. Resize motorcycle to fit scene nicely
# ----------------------------
h_m, w_m = motor.shape[:2]

desired_h = int(h_bg * 0.33)  # about 1/3 of background height
scale = desired_h / float(h_m)
new_w = int(w_m * scale)
new_h = desired_h

motor = cv2.resize(motor, (new_w, new_h), interpolation=cv2.INTER_AREA)
fg_mask = cv2.resize(fg_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

h_m, w_m = motor.shape[:2]

# ----------------------------
# 5. Overlay helper
# ----------------------------
def overlay_with_mask(bg, fg, mask, x, y):
    """
    Overlay fg on bg at (x, y) using mask.
    Returns a new image (bg is not modified in-place).
    """
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = fg.shape[:2]

    # Clip overlay region to background boundaries
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
        return bg.copy()

    roi_bg = bg[bg_y1:bg_y2, bg_x1:bg_x2]
    roi_fg = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    roi_mask = mask[fg_y1:fg_y2, fg_x1:fg_x2]

    alpha = (roi_mask.astype(np.float32) / 255.0)[:, :, None]

    blended = roi_bg.astype(np.float32) * (1.0 - alpha) + \
              roi_fg.astype(np.float32) * alpha

    out = bg.copy()
    out[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)
    return out

# ----------------------------
# 6. Animation loop (fullscreen)
# ----------------------------
y_pos = h_bg - h_m - 10
y_pos = max(0, y_pos)

x_pos = -w_m  # start off-screen to the left

# Speed: pixels per frame
speed = max(5, w_bg // 120)

window_name = "Motorcycle in City (press 'q' or Esc to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame = overlay_with_mask(city, motor, fg_mask, int(x_pos), y_pos)
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(30) & 0xFF  # ~33 fps
    if key == ord('q') or key == 27:  # Esc
        break

    x_pos += speed
    if x_pos > w_bg:
        x_pos = -w_m  # loop back to left

cv2.destroyAllWindows()
