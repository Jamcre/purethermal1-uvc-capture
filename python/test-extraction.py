#!/usr/bin/env python
# -*- coding: utf-8 -*-

from uvctypes import *
import time
import cv2
import numpy as np
try:
    from queue import Queue
except ImportError:
    from queue import Queue
import platform

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
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    data = np.uint8(data)
    data = cv2.applyColorMap(data, current_colormap)
    return data

def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img, f"{val:.1f} degF", loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    x, y = loc
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 10, 1)

def create_colorbar(min_temp, max_temp, height=480, width=100):
    gradient = np.linspace(0, 255, height).astype(np.uint8)
    gradient = np.tile(gradient, (width, 1)).T
    colorbar = cv2.applyColorMap(gradient, current_colormap)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colorbar, f"{max_temp:.1f} degF", (10, 25), font, 0.5, (0,0,0), 2)
    cv2.putText(colorbar, f"{min_temp:.1f} degF", (10, height-15), font, 0.5, (0,0,0), 2)
    
    for i in range(height//5, height, height//5):
        temp = max_temp - (i/height)*(max_temp-min_temp)
        cv2.putText(colorbar, f"{temp:.1f}", (15, i+5), font, 0.4, (0,0,0), 1)
    
    return cv2.copyMakeBorder(colorbar, 0, 0, 3, 3, cv2.BORDER_CONSTANT, value=(255,255,255))

def mouse_callback(event, x, y, flags, param):
    global last_click_pos, last_click_temp, last_click_time
    if event == cv2.EVENT_LBUTTONDOWN and x < 640:
        last_click_pos = (x, y)
        if thermal_data is not None:
            orig_x, orig_y = x//4, y//4
            last_click_temp = thermal_data[orig_y, orig_x]
            last_click_time = time.time()
            print(f"Clicked at ({x}, {y}): {ktof(last_click_temp):.1f} degF")

def main():
    global last_click_pos, last_click_temp, last_click_time, thermal_data, current_colormap
    
    # Initialize last save time
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

            cv2.namedWindow('Lepton Radiometry')
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
                    #print(conv_data[:5, :5])
                    # print("Raw Y16 Data Sample (Top-Left 5x5 pixels):")
                    # print(thermal_data[:5, :5]) 
                    display_data = cv2.resize(data[:,:], (640, 480))
                    map_idx = cv2.getTrackbarPos("Colormap", "Lepton Radiometry")
                    current_colormap = COLORMAPS[map_idx][1]
                    img = raw_to_8bit(display_data)
                    # print("Raw Y16 Data Sample (Top-Left 5x5 pixels):")
                    # print(img[:5, :5])
                    
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data)
                    minLoc = (int(minLoc[0]*4), int(minLoc[1]*4))
                    maxLoc = (int(maxLoc[0]*4), int(maxLoc[1]*4))
                    
                    display_temperature(img, minVal, minLoc, (255, 0, 0))
                    display_temperature(img, maxVal, maxLoc, (0, 0, 255))
                    
                    if last_click_pos is not None and (time.time() - last_click_time) < 3:
                        x, y = last_click_pos
                        display_temperature(img, last_click_temp, (x, y), (0, 255, 0))
                    
                    colorbar = create_colorbar(ktof(maxVal), ktof(minVal))
                    
                    if colorbar.shape[0] != img.shape[0]:
                        colorbar = cv2.resize(colorbar, (colorbar.shape[1], img.shape[0]))
                    
                    display_img = np.hstack((img, colorbar))

                    cv2.putText(img, f"{COLORMAPS[map_idx][0]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    
                    cv2.imshow('Lepton Radiometry', display_img)
                    
                    # Handle key presses and automatic saving
                    current_time = time.time()
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        break
                    elif key == ord('s') or (current_time - last_save_time) >= 30:
                        # Generate timestamp for filenames
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        
                        # Convert raw thermal data to Fahrenheit
                        #raw_fahrenheit = 1.8 * ((thermal_data.astype(np.float32) - 27315) / 100.0 + 32.0)
                        np.savetxt(f"raw_thermal_{timestamp}.csv", thermal_data, delimiter=",")
                        
                        # Convert resized display data to Fahrenheit
                        #resized_fahrenheit = 1.8 * ((display_data.astype(np.float32) - 27315) / 100.0 + 32.0)
                        # ISSUE: 
                        np.savetxt(f"resized_thermal_{timestamp}.csv", conv_data, delimiter=",")
                        
                        print(f"Saved thermal data at {timestamp}")
                        last_save_time = current_time  # Reset save timer

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