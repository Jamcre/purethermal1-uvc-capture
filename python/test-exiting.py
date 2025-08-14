#!/usr/bin/env python
# -*- coding: utf-8 -*-

from uvctypes import *
import time
import cv2
import numpy as np
import os
import sys
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import platform
import subprocess
import re

# ======= Unified Display Configuration =======
class DisplayConfig:
    """Centralized display resolution and scaling for Raspbian/Linux."""
    def __init__(self):
        self.ref_width = 640    # Original development reference
        self.ref_height = 480
        self.taskbar_height = 40  # Raspbian taskbar (adjust if needed)
        self.window_margins = {
            'top': 30,    # Extra space for window decorations
            'left': 0,
            'right': 0,
            'bottom': self.taskbar_height
        }
        self.font_scale_multiplier = 1.8
        self.update_resolution()

    def update_resolution(self):
        """Get actual resolution using xrandr (Linux-only)."""
        try:
            cmd = "xrandr | grep -w connected | awk -F'[ +]' '{print $4}'"
            output = subprocess.check_output(cmd, shell=True).decode()
            res = output.strip().split('x')
            self.raw_width = int(res[0])
            self.raw_height = int(res[1])
            
            # Calculate usable space after accounting for margins
            self.display_width = self.raw_width - (self.window_margins['left'] + self.window_margins['right'])
            self.display_height = self.raw_height - (self.window_margins['top'] + self.window_margins['bottom'])
            
            # Scaling factors (maintaining original 640x480 ratios)
            self.scale_x = self.display_width / self.ref_width
            self.scale_y = self.display_height / self.ref_height
            
            print(f"Detected resolution: {self.raw_width}x{self.raw_height}")
            print(f"Usable area: {self.display_width}x{self.display_height}")
        except Exception as e:
            print(f"Resolution detection failed, using fallback 800x480: {e}")
            self.raw_width = 800
            self.raw_height = 480
            self.display_width = 800 - (self.window_margins['left'] + self.window_margins['right'])
            self.display_height = 480 - (self.window_margins['top'] + self.window_margins['bottom'])
            self.scale_x = self.display_width / self.ref_width
            self.scale_y = self.display_height / self.ref_height

    def scale(self, val, axis='both'):
        """Scale a value proportionally (x, y, or uniform)."""
        if axis == 'x':
            return int(val * self.scale_x)
        elif axis == 'y':
            return int(val * self.scale_y)
        return int(val * min(self.scale_x, self.scale_y))  # uniform

    def font_scale(self, relative_size):
        """Calculate font size with multiplier."""
        return relative_size * self.scale_y * self.font_scale_multiplier

# Initialize display configuration
display = DisplayConfig()

# Constants
THERMAL_WIDTH_RATIO = 0.8
TOP_MARGIN = display.scale(10, 'y')
BUF_SIZE = 2
q = Queue(BUF_SIZE)
EXIT_BUTTON_HEIGHT = display.scale(40, 'y')  # Height of the exit button

# Colormap setup
COLORMAPS = {
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "GRAY": cv2.COLORMAP_BONE
}
current_colormap = COLORMAPS["TURBO"]

# File directories
DIR_RAW = "rawThermalData"
DIR_NORM = "normalisedThermalData"
DIR_IMAGES = "thermalImages"

def create_directories():
    for directory in [DIR_RAW, DIR_NORM, DIR_IMAGES]:
        os.makedirs(directory, exist_ok=True)

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
    norm = cv2.normalize(data, None, 0, 65535, cv2.NORM_MINMAX)
    shifted = np.right_shift(norm, 8).astype(np.uint8)
    colorized = cv2.applyColorMap(shifted, current_colormap)
    return colorized

def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img, f"{val:.1f} degF", loc, cv2.FONT_HERSHEY_SIMPLEX, 
               display.font_scale(0.5), color, thickness())
    x, y = loc
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, marker_size(), thickness())

def marker_size():
    return max(5, display.scale(15, 'x'))

def thickness():
    return max(1, display.scale(1, 'y'))

def create_colorbar(min_temp, max_temp, height=None, width=None):
    if height is None:
        height = display.display_height - EXIT_BUTTON_HEIGHT  # Reserve space for exit button
    if width is None:
        width = display.scale(50, 'x')  # Fixed safe width

    gradient = np.linspace(0, 255, height).astype(np.uint8)
    gradient = np.tile(gradient, (width, 1)).T
    colorbar = cv2.applyColorMap(gradient, current_colormap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = display.font_scale(0.45)
    th = thickness()

    cv2.putText(colorbar, f"{max_temp:.1f} degF", (display.scale(5, 'x'), TOP_MARGIN + display.scale(10, 'y')), 
                font, fs, (0,0,0), th+1)
    cv2.putText(colorbar, f"{min_temp:.1f} degF", (display.scale(5, 'x'), height - TOP_MARGIN), 
                font, fs, (0,0,0), th+1)

    ticks = 5
    for i in range(height//ticks, height, height//ticks):
        temp = max_temp - (i/height)*(max_temp-min_temp)
        cv2.putText(colorbar, f"{temp:.1f}", (display.scale(6, 'x'), i + TOP_MARGIN//2), 
                    font, display.font_scale(0.35), (0,0,0), th)

    # Add exit button at the bottom
    exit_button = np.zeros((EXIT_BUTTON_HEIGHT, width, 3), dtype=np.uint8)
    exit_button[:] = (0, 0, 255)  # Red background
    cv2.putText(exit_button, "EXIT", 
               (width//2 - display.scale(20, 'x'), EXIT_BUTTON_HEIGHT//2 + display.scale(10, 'y')), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               display.font_scale(0.6), 
               (255, 255, 255), 
               thickness())
    
    colorbar = np.vstack((colorbar, exit_button))
    
    return cv2.copyMakeBorder(colorbar, 0, 0, display.scale(3, 'x'), display.scale(3, 'x'), 
                                cv2.BORDER_CONSTANT, value=(255,255,255))

last_click_pos = None
last_click_temp = None
last_click_time = None
thermal_data = None
should_exit = False

def mouse_callback(event, x, y, flags, param):
    global last_click_pos, last_click_temp, last_click_time, should_exit
    
    thermal_img_width = int(display.display_width * THERMAL_WIDTH_RATIO)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < thermal_img_width:
            last_click_pos = (x, y)
            if thermal_data is not None:
                raw_h, raw_w = thermal_data.shape
                scale_x_factor = thermal_img_width / raw_w
                scale_y_factor = display.display_height / raw_h
                orig_x = int(x / scale_x_factor)
                orig_y = int(y / scale_y_factor)
                orig_x = max(0, min(raw_w - 1, orig_x))
                orig_y = max(0, min(raw_h - 1, orig_y))
                last_click_temp = thermal_data[orig_y, orig_x]
                last_click_time = time.time()
                print(f"Clicked at ({x}, {y}): {ktof(last_click_temp):.1f} degF")
        else:
            colorbar_height = display.display_height
            exit_button_top = colorbar_height - EXIT_BUTTON_HEIGHT
            
            if y > exit_button_top:
                print("Exit button clicked")
                should_exit = True

def main():
    global last_click_pos, last_click_temp, last_click_time, thermal_data, current_colormap, should_exit

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

            # Window setup
            cv2.namedWindow('Lepton Radiometry', cv2.WINDOW_NORMAL)
            cv2.moveWindow('Lepton Radiometry', 
                         display.window_margins['left'], 
                         display.window_margins['top'])
            cv2.resizeWindow('Lepton Radiometry', 
                           display.display_width, 
                           display.display_height)
            cv2.setMouseCallback('Lepton Radiometry', mouse_callback)

            colormap_names = list(COLORMAPS.keys())
            cv2.createTrackbar(
                "Colormap",
                "Lepton Radiometry",
                0,
                len(colormap_names) - 1,
                lambda x: None
            )

            try:
                while not should_exit:
                    data = q.get(True, 500)
                    if data is None:
                        break

                    # --- Display and colorbar handling (updated) ---
                    thermal_data = data.copy()
                    conv_data = ktof(thermal_data)

                    thermal_img_height = display.display_height
                    MIN_COLORBAR_WIDTH = display.scale(50, 'x')
                    thermal_img_width = int(display.display_width * THERMAL_WIDTH_RATIO)
                    colorbar_width = display.display_width - thermal_img_width

                    if colorbar_width < MIN_COLORBAR_WIDTH:
                        colorbar_width = MIN_COLORBAR_WIDTH
                        thermal_img_width = display.display_width - colorbar_width

                    display_data = cv2.resize(thermal_data[:, :], (thermal_img_width, thermal_img_height))
                    img = raw_to_8bit(display_data)

                    map_idx = cv2.getTrackbarPos("Colormap", "Lepton Radiometry")
                    current_colormap = COLORMAPS[colormap_names[map_idx]]

                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data)
                    raw_h, raw_w = thermal_data.shape
                    scale_x_factor = thermal_img_width / raw_w
                    scale_y_factor = thermal_img_height / raw_h
                    minLoc = (int(minLoc[0] * scale_x_factor), int(minLoc[1] * scale_y_factor))
                    maxLoc = (int(maxLoc[0] * scale_x_factor), int(maxLoc[1] * scale_y_factor))

                    display_temperature(img, minVal, minLoc, (255, 0, 0))
                    display_temperature(img, maxVal, maxLoc, (0, 0, 255))

                    if last_click_pos is not None and (time.time() - last_click_time) < 3:
                        x, y = last_click_pos
                        display_temperature(img, last_click_temp, (x, y), (0, 255, 0))

                    colorbar = create_colorbar(ktof(maxVal), ktof(minVal), 
                                               height=thermal_img_height, width=colorbar_width)

                    if colorbar.shape[0] != img.shape[0]:
                        colorbar = cv2.resize(colorbar, (colorbar.shape[1], img.shape[0]))

                    display_img = np.hstack((img, colorbar))

                    cv2.putText(
                        img,
                        f"{colormap_names[map_idx]}",
                        (display.scale(10, 'x'), TOP_MARGIN + display.scale(20, 'y')),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        display.font_scale(0.6),
                        (255, 255, 255),
                        thickness()
                    )

                    cv2.imshow('Lepton Radiometry', display_img)
                    cv2.waitKey(1)

                    current_time = time.time()
                    if (current_time - last_save_time) >= 30:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")

                        raw_path = os.path.join(DIR_RAW, f"raw_thermal_{timestamp}.csv")
                        np.savetxt(raw_path, thermal_data, delimiter=",")

                        norm_path = os.path.join(DIR_NORM, f"resized_thermal_{timestamp}.csv")
                        np.savetxt(norm_path, conv_data, delimiter=",")

                        image_path = os.path.join(DIR_IMAGES, f"thermal_image_{timestamp}.png")
                        cv2.imwrite(image_path, display_img)

                        print(f"Saved thermal data at {timestamp}")
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
