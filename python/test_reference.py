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

# Click variables with timeout
last_click_pos = None
last_click_temp = None
last_click_time = None
thermal_data = None  # Will hold the raw thermal data

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
    data = cv2.applyColorMap(data, cv2.COLORMAP_TURBO)
    return data

def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img, "{0:.1f}degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    x, y = loc
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 10, 1)

def create_colorbar(min_temp, max_temp, height=480, width=100):
    """Create a vertical colorbar with min/max labels and professional borders"""
    # Create gradient (original height without borders)
    gradient = np.linspace(0, 255, height).astype(np.uint8)
    gradient = np.tile(gradient, (width, 1)).T
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    text_scale = 0.5
    
    # Max temp label
    (max_text_w, max_text_h), _ = cv2.getTextSize(f"{max_temp:.1f}degF", font, text_scale, text_thickness)
    cv2.rectangle(colorbar, (5,5), (15+max_text_w, 25+max_text_h), (0,0,0), -1)
    cv2.putText(colorbar, f"{max_temp:.1f}degF", (10, 25), 
                font, text_scale, (255,255,255), text_thickness)
    
    # Min temp label
    (min_text_w, min_text_h), _ = cv2.getTextSize(f"{min_temp:.1f}degF", font, text_scale, text_thickness)
    cv2.rectangle(colorbar, (5,height-30), (15+min_text_w, height-10), (0,0,0), -1)
    cv2.putText(colorbar, f"{min_temp:.1f}degF", (10, height-15), 
                font, text_scale, (255,255,255), text_thickness)
    
    # Add scale ticks
    for i in range(height//5, height, height//5): 
        temp = max_temp - (i/height)*(max_temp-min_temp)
        cv2.line(colorbar, (0,i), (10,i), (255,255,255), 1)
        cv2.putText(colorbar, f"{temp:.1f}", (15, i+5), 
                    font, 0.4, (255,255,255), 1)
    
    # Add side borders only (maintain height)
    colorbar = cv2.copyMakeBorder(colorbar, 0, 0, 3, 3, cv2.BORDER_CONSTANT, value=(255,255,255))
    colorbar = cv2.copyMakeBorder(colorbar, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    return colorbar

def mouse_callback(event, x, y, flags, param):
    global last_click_pos, last_click_temp, last_click_time
    if event == cv2.EVENT_LBUTTONDOWN and x < 640:  # Only accept clicks on thermal image
        last_click_pos = (x, y)
        if thermal_data is not None:
            orig_x, orig_y = x//4, y//4
            last_click_temp = thermal_data[orig_y, orig_x]
            last_click_time = time.time()
            print(f"Clicked at ({x}, {y}): {ktof(last_click_temp):.1f}degF / {ktoc(last_click_temp):.1f}degC")

def main():
    global last_click_pos, last_click_temp, last_click_time, thermal_data
    
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

            try:
                while True:
                    data = q.get(True, 500)
                    if data is None:
                        break
                        
                    thermal_data = data.copy()
                    display_data = cv2.resize(data[:,:], (640, 480))
                    img = raw_to_8bit(display_data)
                    
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data)
                    minLoc = (int(minLoc[0]*4), int(minLoc[1]*4))
                    maxLoc = (int(maxLoc[0]*4), int(maxLoc[1]*4))
                    
                    display_temperature(img, minVal, minLoc, (255, 0, 0))
                    display_temperature(img, maxVal, maxLoc, (0, 0, 255))
                    
                    # Show green marker only if clicked within last 3 seconds
                    if last_click_pos is not None and (time.time() - last_click_time) < 3:
                        x, y = last_click_pos
                        display_temperature(img, last_click_temp, (x, y), (0, 255, 0))
                    
                    # Create colorbar and combine with thermal image
                    colorbar = create_colorbar(ktof(maxVal), ktof(minVal))
                    
                    # Resize colorbar to match image height if needed
                    if colorbar.shape[0] != img.shape[0]:
                        colorbar = cv2.resize(colorbar, (colorbar.shape[1], img.shape[0]))
                    
                    # Combine images
                    display_img = np.hstack((img, colorbar))
                    
                    cv2.imshow('Lepton Radiometry', display_img)
                    if cv2.waitKey(1) == 27:
                        break

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