#!/usr/bin/env python
# -*- coding: utf-8 -*-

from uvctypes import *
import time
import cv2
import numpy as np
import os
try:
    from queue import Queue
except ImportError:
    from queue import Queue
import platform

# =========================
# ADAPTIVE DISPLAY SETTINGS
# =========================
# Set your display resolution here (change for different screens)
DISPLAY_WIDTH = 480   # e.g. Raspberry Pi HAT width
DISPLAY_HEIGHT = 320  # e.g. Raspberry Pi HAT height

# Fraction of display width reserved for thermal image (rest used by colorbar)
THERMAL_WIDTH_RATIO = 0.8

# Reference baseline (used only for relative scaling of fonts/markers)
# This keeps look similar to original when DISPLAY is 640x480 baseline
REF_WIDTH = 640
REF_HEIGHT = 480

def scale_x(val):
    return int(val * (DISPLAY_WIDTH / REF_WIDTH))

def scale_y(val):
    return int(val * (DISPLAY_HEIGHT / REF_HEIGHT))

def font_scale(relative):
    # relative is the font size used on REF_HEIGHT baseline (e.g., 0.5)
    return relative * (DISPLAY_HEIGHT / REF_HEIGHT)

def thickness():
    return max(1, scale_y(1))

def marker_size():
    # marker length (original was 10)
    return max(3, scale_x(10))

# ================
# Original globals
# ================
BUF_SIZE = 2
q = Queue(BUF_SIZE)

COLORMAPS = [
    ("TURBO", cv2.COLORMAP_TURBO),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("JET", cv2.COLORMAP_JET),
    ("HOT", cv2.COLORMAP_HOT),
    ("GRAY", cv2.COLORMAP_BONE)
]
current_colormap = cv2.COLORMAP_TURBO

last_click_pos = None
last_click_temp = None
last_click_time = None
thermal_data = None

DIR_RAW = "rawThermalData"
DIR_NORM = "normalisedThermalData"
DIR_IMAGES = "thermalImages"

def create_directories():
    for directory in [DIR_RAW, DIR_NORM, DIR_IMAGES]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(frame.contents.height, frame.contents.width)
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return
    if not q.full():
        q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
    return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
    return (val - 27315) / 100.0

def raw_to_8bit(data):
    # Do NOT normalize in-place to avoid altering original thermal_data
    norm = cv2.normalize(data, None, 0, 65535, cv2.NORM_MINMAX)
    # shift down to 8-bit
    shifted = np.right_shift(norm, 8).astype(np.uint8)
    colorized = cv2.applyColorMap(shifted, current_colormap)
    return colorized

def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    fs = font_scale(0.5)
    th = thickness()
    # putText expects a float font scale; ensure it's not zero
    cv2.putText(img, f"{val:.1f} degF", loc, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)
    x, y = loc
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, marker_size(), th)

def create_colorbar(min_temp, max_temp, height=None, width=None):
    # Default to display height and remaining width
    if height is None:
        height = DISPLAY_HEIGHT
    if width is None:
        width = max(40, int(DISPLAY_WIDTH * (1 - THERMAL_WIDTH_RATIO)))

    gradient = np.linspace(0, 255, height).astype(np.uint8)
    gradient = np.tile(gradient, (width, 1)).T
    colorbar = cv2.applyColorMap(gradient, current_colormap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = font_scale(0.45)
    th = max(1, scale_y(1))

    # top / bottom labels with scaled positions
    cv2.putText(colorbar, f"{max_temp:.1f} degF", (scale_x(5), scale_y(20)), font, fs, (0,0,0), th+1)
    cv2.putText(colorbar, f"{min_temp:.1f} degF", (scale_x(5), height - scale_y(10)), font, fs, (0,0,0), th+1)

    # intermediate ticks
    ticks = 5
    for i in range(height//ticks, height, height//ticks):
        temp = max_temp - (i/height)*(max_temp-min_temp)
        cv2.putText(colorbar, f"{temp:.1f}", (scale_x(6), i + scale_y(4)), font, font_scale(0.35), (0,0,0), th)

    return cv2.copyMakeBorder(colorbar, 0, 0, scale_x(3), scale_x(3), cv2.BORDER_CONSTANT, value=(255,255,255))

def mouse_callback(event, x, y, flags, param):
    global last_click_pos, last_click_temp, last_click_time
    # Use DISPLAY_WIDTH instead of hardcoded 640
    if event == cv2.EVENT_LBUTTONDOWN and x < DISPLAY_WIDTH:
        last_click_pos = (x, y)
        if thermal_data is not None:
            # compute scale mapping given current display and thermal array size
            raw_h, raw_w = thermal_data.shape
            thermal_img_width = int(DISPLAY_WIDTH * THERMAL_WIDTH_RATIO)
            scale_x_factor = thermal_img_width / raw_w
            scale_y_factor = DISPLAY_HEIGHT / raw_h
            orig_x = int(x / scale_x_factor)
            orig_y = int(y / scale_y_factor)
            # clamp to valid indexes
            orig_x = max(0, min(raw_w - 1, orig_x))
            orig_y = max(0, min(raw_h - 1, orig_y))
            last_click_temp = thermal_data[orig_y, orig_x]
            last_click_time = time.time()
            print(f"Clicked at ({x}, {y}): {ktof(last_click_temp):.1f} degF")

def main():
    global last_click_pos, last_click_temp, last_click_time, thermal_data, current_colormap

    create_directories()

    last_save_time = time.time()

    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        print("uvc_init error")
        exit(1)

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)

        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0:
                print("uvc_open error")
                exit(1)

            print("device opened!")

            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
            )

            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print("uvc_start_streaming failed: {0}".format(res))
                exit(1)

            # Create a resizable window sized to the target display
            cv2.namedWindow('Lepton Radiometry', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Lepton Radiometry', DISPLAY_WIDTH, DISPLAY_HEIGHT)
            cv2.setMouseCallback('Lepton Radiometry', mouse_callback)

            cv2.createTrackbar(
                "Colormap",
                "Lepton Radiometry",
                0,
                len(COLORMAPS)-1,
                lambda x: None
            )

            try:
                while True:
                    data = q.get(True, 500)
                    if data is None:
                        break

                    thermal_data = data.copy()
                    conv_data = ktof(thermal_data)

                    # Determine adaptive thermal image size (leave space for colorbar)
                    thermal_img_width = int(DISPLAY_WIDTH * THERMAL_WIDTH_RATIO)
                    thermal_img_height = DISPLAY_HEIGHT

                    # Resize raw thermal to display area for visualization
                    display_data = cv2.resize(data[:, :], (thermal_img_width, thermal_img_height))

                    map_idx = cv2.getTrackbarPos("Colormap", "Lepton Radiometry")
                    current_colormap = COLORMAPS[map_idx][1]
                    img = raw_to_8bit(display_data)

                    # Find min/max in original raw array (not the resized one)
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data)

                    # Map raw min/max locations to display coords using scale factors
                    raw_h, raw_w = thermal_data.shape
                    scale_x_factor = thermal_img_width / raw_w
                    scale_y_factor = thermal_img_height / raw_h
                    minLoc = (int(minLoc[0] * scale_x_factor), int(minLoc[1] * scale_y_factor))
                    maxLoc = (int(maxLoc[0] * scale_x_factor), int(maxLoc[1] * scale_y_factor))

                    display_temperature(img, minVal, minLoc, (255, 0, 0))
                    display_temperature(img, maxVal, maxLoc, (0, 0, 255))

                    if last_click_pos is not None and (time.time() - last_click_time) < 3:
                        x, y = last_click_pos
                        # If the click was in the area of the thermal image, show marker
                        display_temperature(img, last_click_temp, (x, y), (0, 255, 0))

                    # Create scaled colorbar to match thermal image height
                    colorbar_width = DISPLAY_WIDTH - thermal_img_width
                    if colorbar_width < scale_x(40):
                        # Ensure minimum width for readability
                        colorbar_width = scale_x(40)
                        # if colorbar pushes total width > DISPLAY_WIDTH, reduce thermal_img_width
                        if thermal_img_width + colorbar_width > DISPLAY_WIDTH:
                            thermal_img_width = DISPLAY_WIDTH - colorbar_width
                            img = cv2.resize(raw_to_8bit(cv2.resize(data[:, :], (thermal_img_width, thermal_img_height))),
                                             (thermal_img_width, thermal_img_height))

                    colorbar = create_colorbar(ktof(maxVal), ktof(minVal), height=thermal_img_height, width=colorbar_width)

                    # Ensure colorbar height matches image height
                    if colorbar.shape[0] != img.shape[0]:
                        colorbar = cv2.resize(colorbar, (colorbar.shape[1], img.shape[0]))

                    display_img = np.hstack((img, colorbar))

                    # Put current colormap name on the thermal image (scaled positions and fonts)
                    cv2.putText(img, f"{COLORMAPS[map_idx][0]}", (scale_x(10), scale_y(30)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale(0.6), (255,255,255), thickness())

                    cv2.imshow('Lepton Radiometry', display_img)

                    current_time = time.time()
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    elif key == ord('s') or (current_time - last_save_time) >= 30:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")

                        raw_path = os.path.join(DIR_RAW, f"raw_thermal_{timestamp}.csv")
                        np.savetxt(raw_path, thermal_data, delimiter=",")

                        norm_path = os.path.join(DIR_NORM, f"resized_thermal_{timestamp}.csv")
                        np.savetxt(norm_path, conv_data, delimiter=",")

                        image_path = os.path.join(DIR_IMAGES, f"thermal_image_{timestamp}.png")
                        cv2.imwrite(image_path, display_img)

                        print(f"Saved thermal data at {timestamp}")
                        print(f"  Raw data: {raw_path}")
                        print(f"  Normalized data: {norm_path}")
                        print(f"  Image: {image_path}")
                        last_save_time = current_time

                cv2.destroyAllWindows()
            finally:
                libuvc.uvc_stop_streaming(devh)

            print("done")
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)

if __name__ == '__main__':
    main()
