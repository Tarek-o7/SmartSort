import cv2
import numpy as np
import time
import random
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import threading

# Camera config
CAM_WIDTH = 2304
CAM_HEIGHT = 1296
CROP_Y = 400
CROP_H = 300

MIN_RADIUS = 120
MAX_RADIUS = 150

OFF_MS = 100
SNAPSHOT_MS = 50

global_next_id = 1
active_ids = []
classification = {}
lost_apple_info = {}
lost_popups = {}
POPUP_DURATION = 1.0

# Servo Setup
SERVO_PIN = 4
POS_ROTTEN     = 20
POS_BLEMISHED  = 33
POS_STORE_READY= 47
last_servo_angle = None

def angle_to_duty(angle):
    duty = (angle / 18.0) + 2.5
    return max(3.0, min(11.0, duty))

def move_servo_async(pwm, angle):
    global last_servo_angle
    if last_servo_angle == angle:
        print(f"[SERVO] Already at angle {angle}, skipping movement.")
        return
    print(f"[SERVO] Moving to angle {angle}")
    duty = angle_to_duty(angle)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1.5)
    pwm.ChangeDutyCycle(0)
    last_servo_angle = angle

# Snapshot Gallery
gallery_height = 100
MAX_COLUMNS = 6
snapshot_gallery_rows = {}

def update_gallery(new_snap, apple_id):
    h, w = new_snap.shape[:2]
    scale = gallery_height / h
    resized = cv2.resize(new_snap, (int(w * scale), gallery_height))

    if apple_id not in snapshot_gallery_rows:
        if len(snapshot_gallery_rows) >= MAX_COLUMNS:
            oldest_id = sorted(snapshot_gallery_rows.keys())[0]
            del snapshot_gallery_rows[oldest_id]
        snapshot_gallery_rows[apple_id] = []

    snapshot_gallery_rows[apple_id].append(resized)

    if len(snapshot_gallery_rows[apple_id]) > 7:
        snapshot_gallery_rows[apple_id].pop(0)

    rows = []
    max_width = 0
    temp_rows = {}

    for aid in sorted(snapshot_gallery_rows.keys()):
        snaps = snapshot_gallery_rows[aid]
        if not snaps:
            continue
        row_img = cv2.hconcat(snaps)
        temp_rows[aid] = row_img
        if row_img.shape[1] > max_width:
            max_width = row_img.shape[1]

    for aid in sorted(temp_rows.keys()):
        row_img = temp_rows[aid]
        (h2, w2, c2) = row_img.shape
        if w2 < max_width:
            pad = np.full((h2, max_width - w2, c2), 255, dtype=row_img.dtype)
            row_img = cv2.hconcat([row_img, pad])
        rows.append(row_img)

    if rows:
        gallery = cv2.vconcat(rows)
        cv2.imshow("Snapshot Gallery", gallery)

def classify_snapshot(snapshot):
    hsv = cv2.cvtColor(snapshot, cv2.COLOR_BGR2HSV)
    total_pixels = snapshot.shape[0] * snapshot.shape[1]

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 70, 140])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    dark_pixels = cv2.countNonZero(mask_dark)
    dark_ratio = dark_pixels / total_pixels
    if dark_ratio >= 0.011:
        return "Rotten"

    lower_orange = np.array([10, 80, 120])
    upper_orange = np.array([45, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    orange_pixels = cv2.countNonZero(mask_orange)
    orange_ratio = orange_pixels / total_pixels
    if orange_ratio >= 0.0135:
        return "Blemished"

    return "Store-Ready"

def detect_red_apples(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([165, 150, 130])
    upper_red1 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([38, 0, 3])
    upper_red2 = np.array([110, 18, 53])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    combined = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if MIN_RADIUS <= radius <= MAX_RADIUS:
            centers.append((int(x), int(y), int(radius)))
    return centers

def draw_section_lines(frame):
    NUM_SECTIONS = 6
    SECTION_OFFSET = 175
    h, w = frame.shape[:2]
    section_width = w / NUM_SECTIONS

    for i in range(1, NUM_SECTIONS):
        x = int(i * section_width + SECTION_OFFSET)
        cv2.line(frame, (x, 0), (x, h), (0, 0, 0), 2)

    for i in range(NUM_SECTIONS):
        label = f"Section {i}"
        x_text = int(i * section_width + SECTION_OFFSET) + 10 if i != 0 else 10
        y_text = 30
        cv2.putText(frame, label, (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3)
        cv2.putText(frame, label, (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

def draw_lost_popups(frame):
    current_time = time.time()
    y_offset = 30
    for apple_id, (message, start_time) in list(lost_popups.items()):
        elapsed = current_time - start_time
        if elapsed < POPUP_DURATION:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thick = 2
            ts, _ = cv2.getTextSize(message, font, font_scale, thick)
            x = (frame.shape[1] - ts[0]) // 2
            y = y_offset
            y_offset += ts[1] + 15
            cv2.rectangle(frame, (x-5, y-ts[1]-5), (x+ts[0]+5, y+5), (255,255,255), -1)
            cv2.putText(frame, message, (x, y), font, font_scale, (0,0,255), thick)
        else:
            del lost_popups[apple_id]

def display_live_log():
    log_img = np.full((400, 600, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    y_offset = 20

    lines = ["Active Apples:"]
    for aid in sorted(active_ids):
        c = classification.get(aid, "Store-Ready")
        lines.append(f"ID {aid}: {c}")

    lines.append("")
    lines.append("Lost Apples:")
    lost_keys = sorted(lost_apple_info.keys())
    if len(lost_keys) > 7:
        lost_keys = lost_keys[-7:]
    for aid in lost_keys:
        c = lost_apple_info[aid]
        lines.append(f"ID {aid}: {c}")

    for i, line in enumerate(lines):
        y = y_offset + i * line_height
        cv2.putText(log_img, line, (10, y), font, font_scale, (0,0,0), thickness)

    cv2.imshow("Apple Log", log_img)

def main():
    global global_next_id, active_ids, classification, lost_popups, lost_apple_info

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    servo_pwm = GPIO.PWM(SERVO_PIN, 50)
    servo_pwm.start(0)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    window_name = "Ephemeral + 3-Class (Rotten/Blemished/Store-Ready) + Log + SERVO"
    cv2.namedWindow(window_name)

    viewer_started = False
    last_refresh_time = 0.0
    frozen_frame = None

    standby = True
    while standby:
        frame = picam2.capture_array()
        cropped = frame[CROP_Y:CROP_Y + CROP_H, :]

        msg = "Press 's' to start capturing"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        ts, _ = cv2.getTextSize(msg, font, scale, thick)
        text_x = (cropped.shape[1] - ts[0]) // 2
        text_y = cropped.shape[0] // 2
        cv2.putText(cropped, msg, (text_x, text_y), font, scale, (0,0,255), thick)

        cv2.imshow(window_name, cropped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            viewer_started = True
            last_refresh_time = time.time()
            standby = False
        elif key == ord('q'):
            servo_pwm.stop()
            GPIO.cleanup()
            picam2.stop()
            cv2.destroyAllWindows()
            return

    while True:
        frame = picam2.capture_array()
        cropped = frame[CROP_Y:CROP_Y + CROP_H, :].copy()

        centers = detect_red_apples(cropped)
        centers.sort(key=lambda c: c[0])
        N = len(centers)
        current_pool_size = len(active_ids)

        if N > current_pool_size:
            needed = N - current_pool_size
            for _ in range(needed):
                new_id = global_next_id
                active_ids.append(new_id)
                classification[new_id] = "Store-Ready"
                global_next_id += 1
            active_ids.sort()
        elif N < current_pool_size:
            remove_count = current_pool_size - N
            active_ids.sort()
            removed_ids = active_ids[:remove_count]
            active_ids = active_ids[remove_count:]

            for rid in removed_ids:
                label_class = classification.get(rid, "Store-Ready")
                lost_popups[rid] = (f"Apple ID {rid} is {label_class}, left", time.time())

                if label_class == "Rotten":
                    move_servo_async(servo_pwm, POS_ROTTEN)
                elif label_class == "Blemished":
                    move_servo_async(servo_pwm, POS_BLEMISHED)
                else:
                    move_servo_async(servo_pwm, POS_STORE_READY)

                lost_apple_info[rid] = label_class
                if len(lost_apple_info) > 7:
                    oldest_lost = sorted(lost_apple_info.keys())[0]
                    del lost_apple_info[oldest_lost]

        for i, (cx, cy, r) in enumerate(centers):
            apple_id = active_ids[i]
            label_class = classification.get(apple_id, "Store-Ready")
            text_label = f"ID {apple_id} ({label_class})"
            if label_class == "Rotten":
                color = (0, 0, 0)
            elif label_class == "Blemished":
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.circle(cropped, (cx, cy), 5, color, -1)
            cv2.putText(cropped, text_label, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.circle(cropped, (cx, cy), r, color, 2)

            now = time.time()
            elapsed_ms = (now - last_refresh_time) * 1000.0
            snapshot_window_start = OFF_MS - (SNAPSHOT_MS / 2.0)
            snapshot_window_end   = OFF_MS + (SNAPSHOT_MS / 2.0)

            if snapshot_window_start <= elapsed_ms <= snapshot_window_end:
                sub_size = (160, 160)
                sub_img = cv2.getRectSubPix(cropped, sub_size, (cx, cy))
                update_gallery(sub_img, apple_id)

                snap_class = classify_snapshot(sub_img)
                if snap_class == "Rotten":
                    classification[apple_id] = "Rotten"
                elif snap_class == "Blemished":
                    if classification[apple_id] != "Rotten":
                        classification[apple_id] = "Blemished"

        draw_section_lines(cropped)
        draw_lost_popups(cropped)

        now = time.time()
        elapsed_ms = (now - last_refresh_time) * 1000.0
        if viewer_started and elapsed_ms >= OFF_MS:
            frozen_frame = cropped.copy()
            last_refresh_time = now

        if frozen_frame is not None:
            cv2.imshow(window_name, frozen_frame)
        else:
            cv2.imshow(window_name, cropped)

        display_live_log()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    servo_pwm.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()