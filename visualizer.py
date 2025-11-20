import cv2
import numpy as np
import time

class BoxSmoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.tracks = {} # {track_id: [x1, y1, x2, y2]}

    def update(self, track_id, coords):
        if track_id not in self.tracks:
            self.tracks[track_id] = coords
            return coords
        
        # Exponential Moving Average
        prev = self.tracks[track_id]
        new_coords = []
        for p, n in zip(prev, coords):
            new_coords.append(int(self.alpha * n + (1 - self.alpha) * p))
        
        self.tracks[track_id] = new_coords
        return new_coords

class Visualizer:
    def __init__(self):
        self.fps_history = []
        self.font = cv2.FONT_HERSHEY_DUPLEX # Cleaner font
        self.smoother = BoxSmoother(alpha=0.4)
        self.notifications = [] # List of (text, start_time)

    def add_notification(self, text):
        self.notifications.append((text, time.time()))

    def draw_detections(self, frame, detections, detector):
        # Create a SINGLE overlay for all transparency effects
        # This fixes the "flashing" caused by multiple addWeighted calls
        overlay = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = detector.get_color(det['id'])
            label = det['class']
            conf = det['conf']
            track_id = det['id']
            
            # Smooth the box
            if track_id is not None:
                x1, y1, x2, y2 = self.smoother.update(track_id, [x1, y1, x2, y2])
            
            # Draw filled box on overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Draw corners on MAIN frame (solid)
            length = (x2 - x1) // 4
            thickness = 2
            # Top-Left
            cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
            # Top-Right
            cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
            # Bottom-Left
            cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
            # Bottom-Right
            cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

            # Label
            label_text = f"{label} {int(conf*100)}%"
            if track_id is not None:
                label_text = f"ID:{track_id} {label_text}"
            
            (w, h), _ = cv2.getTextSize(label_text, self.font, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 8), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Apply transparency ONCE
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw_ui(self, frame, fps, model_name, device, mode='ALL'):
        # FPS Graph
        self.fps_history.append(fps)
        if len(self.fps_history) > 50:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        h, w = frame.shape[:2]
        
        # Glassmorphism Status Bar (Smaller & Cleaner)
        bar_w = 500
        bar_h = 30
        bar_x = (w - bar_w) // 2
        bar_y = 10
        
        # Draw UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status Text
        status_text = f"FPS: {int(avg_fps)} | {model_name} | {device.upper()} | Mode: {mode}"
        text_size = cv2.getTextSize(status_text, self.font, 0.5, 1)[0]
        text_x = bar_x + (bar_w - text_size[0]) // 2
        text_y = bar_y + 20
        
        cv2.putText(frame, status_text, (text_x, text_y), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Controls Hint (Bottom)
        hint_text = "[1-3] Model | [T] Track | [C] Custom | [M] Mode | [Q] Quit"
        cv2.putText(frame, hint_text, (20, h - 20), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Draw Notifications
        self.draw_notifications(frame)

    def draw_notifications(self, frame):
        current_time = time.time()
        # Filter old notifications (display for 2 seconds)
        self.notifications = [(text, t) for text, t in self.notifications if current_time - t < 2.0]
        
        h, w = frame.shape[:2]
        
        for i, (text, t) in enumerate(self.notifications):
            # Fade out effect
            age = current_time - t
            
            # Center text
            font_scale = 1.0
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
            tx = (w - tw) // 2
            ty = (h // 2) + (i * 40)
            
            # Draw with outline for visibility
            cv2.putText(frame, text, (tx, ty), self.font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty), self.font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def draw_input_box(self, frame, current_text):
        h, w = frame.shape[:2]
        
        # Darken background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Input Box
        box_w = 800
        box_h = 100
        box_x = (w - box_w) // 2
        box_y = (h - box_h) // 2
        
        # Draw Box
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ENTER CUSTOM CLASSES", (box_x + 20, box_y - 20), self.font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Text
        display_text = current_text + "|" # Cursor
        cv2.putText(frame, display_text, (box_x + 20, box_y + 65), self.font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Instructions
        cv2.putText(frame, "Type names separated by commas. Press ENTER to confirm, ESC to cancel.", (box_x, box_y + box_h + 30), self.font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
