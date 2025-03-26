import cv2
import numpy as np
import pygame

# Initialize pygame with sufficient audio channels
pygame.mixer.init()  # Increased channel count
pygame.init()

# Create channel objects for simultaneous playback
CHANNELS = [pygame.mixer.Channel(i) for i in range(8)]  # Match the initialized channel count

drum_sounds = {
    "snare": pygame.mixer.Sound("Audio/Snare.WAV"),
    "floor_tom": pygame.mixer.Sound("Audio/Floor.WAV"),
    "high_tom": pygame.mixer.Sound("Audio/High-tom.WAV"),
    "Low_tom": pygame.mixer.Sound("Audio/Low-tom.WAV"),
    "hi_hat": pygame.mixer.Sound("Audio/Hi-hat.WAV"),
    "ride_cymbal": pygame.mixer.Sound("Audio/Ride.WAV"),
    "crash_cymbal": pygame.mixer.Sound("Audio/Crash.WAV")
}

# Global variables
drum_zones = {}  # (center, radius, mask, area)
selected_zone = None
current_drum = None
selecting = False
radius = 0
stick_colors = {"left": None, "right": None}
setting_drum_areas = True
frame = None
prev_hits = {}
kernel = np.ones((5,5), np.uint8)

def mouse_callback(event, x, y, flags, param):
    global selected_zone, current_drum, selecting, radius, setting_drum_areas, stick_colors, frame
    
    if setting_drum_areas:
        if event == cv2.EVENT_LBUTTONDOWN and current_drum:
            selected_zone = (x, y)
            selecting = True
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            radius = int(((x - selected_zone[0])**2 + (y - selected_zone[1])**2)**0.5)
        elif event == cv2.EVENT_LBUTTONUP and current_drum:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.circle(mask, selected_zone, radius, 255, -1)
            area = np.pi * radius**2
            drum_zones[current_drum] = (selected_zone, radius, mask, area)
            prev_hits[current_drum] = False
            selecting = False
            radius = 0
            if len(drum_zones) == len(drum_sounds):
                setting_drum_areas = False
            else:
                next_drum()
    else:
        if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color = hsv[y, x]
            lower_bound = np.maximum(color - np.array([20, 100, 100]), 0)
            upper_bound = np.minimum(color + np.array([20, 255, 255]), 255)
            if stick_colors["left"] is None:
                stick_colors["left"] = (lower_bound, upper_bound)
            elif stick_colors["right"] is None:
                stick_colors["right"] = (lower_bound, upper_bound)

def next_drum():
    global current_drum
    drum_list = list(drum_sounds.keys())
    index = drum_list.index(current_drum)
    if index < len(drum_list) - 1:
        current_drum = drum_list[index + 1]

def detect_hit(frame, hsv):
    if setting_drum_areas or None in stick_colors.values():
        return
    
    left_mask = cv2.inRange(hsv, *stick_colors["left"])
    right_mask = cv2.inRange(hsv, *stick_colors["right"])
    
    left_mask = cv2.morphologyEx(left_mask, cv2.MORPH_OPEN, kernel)
    right_mask = cv2.morphologyEx(right_mask, cv2.MORPH_OPEN, kernel)
    
    # Collect all hits first before playing sounds
    hits_to_play = []
    
    for drum, (center, radius, mask, area) in drum_zones.items():
        threshold = area * 0.05
        
        left_roi = cv2.bitwise_and(left_mask, left_mask, mask=mask)
        left_pixels = cv2.countNonZero(left_roi)
        
        right_roi = cv2.bitwise_and(right_mask, right_mask, mask=mask)
        right_pixels = cv2.countNonZero(right_roi)
        
        if (left_pixels > threshold or right_pixels > threshold) and not prev_hits[drum]:
            hits_to_play.append(drum)
            prev_hits[drum] = True
        elif left_pixels < threshold/2 and right_pixels < threshold/2:
            prev_hits[drum] = False
    
    # Play all detected hits using available channels
    for drum in hits_to_play:
        for channel in CHANNELS:
            if not channel.get_busy():
                channel.play(drum_sounds[drum])
                break

cap = cv2.VideoCapture(1)
cv2.namedWindow("Drum Setup")
cv2.setMouseCallback("Drum Setup", mouse_callback)

drum_list = list(drum_sounds.keys())
current_drum = drum_list[0]

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for drum, (center, radius, _, _) in drum_zones.items():
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.putText(frame, drum, (center[0]-20, center[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        if selecting and selected_zone:
            cv2.circle(frame, selected_zone, radius, (255,0,0), 2)
        
        detect_hit(frame, hsv)
        
        status_text = "Select: {}".format(current_drum) if setting_drum_areas else \
                      "Click to select drumstick colors" if None in stick_colors.values() else \
                      "Playing mode"
        color = (255, 0, 0) if setting_drum_areas else \
                (0, 0, 255) if None in stick_colors.values() else \
                (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Drum Setup", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()