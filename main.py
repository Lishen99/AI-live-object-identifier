import cv2
import time
import argparse
import numpy as np
from camera import Camera
from detector import Detector
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description='AI Object Detector V6 (Professional)')
    parser.add_argument('--model', type=str, default='yolov8x-world.pt', help='Initial model to load')
    args = parser.parse_args()

    # Initialize modules
    print("Initializing Camera...")
    cam = Camera(src=0)
    
    print("Initializing Detector...")
    detector = Detector(model_name=args.model)
    
    # Auto-load Master Vocabulary if using World model
    if 'world' in args.model:
        detector.load_master_vocabulary()
    
    print("Initializing Visualizer...")
    viz = Visualizer()

    print("Starting Main Loop.")
    print("Controls: [1-3] Switch Models | [T] Toggle Tracking | [C] Custom Classes | [M] Switch Mode | [Q] Quit")
    
    # Create window and set to maximized (Windowed Fullscreen behavior)
    window_name = 'AI Object Detector V7'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Removed for V7
    # Maximize window
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) # Keep on top initially
    
    prev_time = 0
    
    # Simplified Model mapping (3 Tiers)
    models = {
        ord('1'): 'yolov8n.pt',       # Low (Pi/CPU)
        ord('2'): 'yolov8s-world.pt', # Medium (Laptop/Low GPU)
        ord('3'): 'yolov8x-world.pt'  # High (RTX/Desktop)
    }
    
    try:
        while True:
            frame = cam.get_frame()
            
            if frame is not None:
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time
                
                # Run detection
                detections = detector.detect(frame)
                
                # Visualize (Single Pass)
                viz.draw_detections(frame, detections, detector)
                
                # Draw Stats & Notifications
                viz.draw_ui(frame, fps, detector.model_name, detector.device, detector.mode)
                
                # --- Letterboxing (Aspect Ratio Correction) ---
                # Get current window size
                try:
                    win_rect = cv2.getWindowImageRect(window_name)
                    screen_w, screen_h = win_rect[2], win_rect[3]
                except:
                    screen_w, screen_h = 1920, 1080 # Fallback
                
                if screen_w <= 0 or screen_h <= 0: # Safety
                    screen_w, screen_h = 1920, 1080

                # Calculate scaling to fit
                h, w = frame.shape[:2]
                scale = min(screen_w / w, screen_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize frame
                resized_frame = cv2.resize(frame, (new_w, new_h))
                
                # Create black canvas
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                
                # Center the frame
                x_offset = (screen_w - new_w) // 2
                y_offset = (screen_h - new_h) // 2
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
                
                # Show canvas
                cv2.imshow(window_name, canvas)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                state = detector.toggle_tracking()
                status = 'Enabled' if state else 'Disabled'
                print(f"Tracking {status}")
                viz.add_notification(f"Tracking {status}")
            elif key == ord('m'):
                modes = ['ALL', 'LIVING', 'OBJECTS']
                current_index = modes.index(detector.mode)
                new_mode = modes[(current_index + 1) % len(modes)]
                detector.set_mode(new_mode)
                print(f"Switched Mode to: {new_mode}")
                viz.add_notification(f"Mode: {new_mode}")
            elif key == ord('c'):
                # Enter Input Mode
                input_mode = True
                user_text = ""
                viz.add_notification("Type Custom Classes...")
                
                while input_mode:
                    # Draw the last frame with the input box overlay
                    # We need to keep the camera running or at least show the last frame
                    # For simplicity, we'll just redraw the last frame + input box
                    # But to keep it "live", we can fetch new frames if we want.
                    # Let's keep it live!
                    
                    frame = cam.get_frame()
                    if frame is None: break
                    
                    # Resize/Letterbox logic (reused)
                    h, w = frame.shape[:2]
                    scale = min(screen_w / w, screen_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    x_offset = (screen_w - new_w) // 2
                    y_offset = (screen_h - new_h) // 2
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
                    
                    # Draw Input Box on Canvas
                    viz.draw_input_box(canvas, user_text)
                    cv2.imshow(window_name, canvas)
                    
                    # Wait for key
                    key_input = cv2.waitKey(1) & 0xFF
                    
                    if key_input == 255: continue # No key pressed
                    
                    if key_input == 13: # ENTER
                        if user_text.strip():
                            classes = [c.strip() for c in user_text.split(',')]
                            detector.set_classes(classes)
                            viz.add_notification(f"Custom: {len(classes)} Classes")
                        input_mode = False
                    elif key_input == 27: # ESC
                        input_mode = False
                        viz.add_notification("Cancelled Input")
                    elif key_input == 8: # BACKSPACE
                        user_text = user_text[:-1]
                    elif 32 <= key_input <= 126: # Printable chars
                        user_text += chr(key_input)
            
            # Model Switching
            if key in models:
                model_name = models[key]
                viz.add_notification(f"Loading {model_name}...")
                cv2.imshow(window_name, canvas) 
                cv2.waitKey(1)
                
                detector.load_model(model_name)
                if 'world' in model_name:
                    detector.load_master_vocabulary()
                viz.add_notification(f"Loaded {model_name}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
