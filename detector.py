from ultralytics import YOLO
import cv2
import numpy as np
import random
import torch

class Detector:
    def __init__(self, model_name='yolov8s-world.pt'):
        # Cross-platform device detection
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        print(f"Using device: {self.device}")
        self.filter_classes = None # For standard YOLO models
        self.load_model(model_name)
        self.colors = {}
        self.tracking_enabled = True # Default to True
        self.mode = 'ALL' # ALL, LIVING, OBJECTS
        self.master_vocab = []

    def load_model(self, model_name):
        self.model_name = model_name
        print(f"Loading model: {model_name}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.filter_classes = None # Reset filter on load
        print("Model loaded.")

    def set_classes(self, class_list):
        print(f"Setting custom classes: {class_list}")
        
        # Check if it's a YOLO-World model (has set_classes)
        if hasattr(self.model, 'set_classes'):
            # Check if model is currently in Half precision
            is_half = False
            try:
                if hasattr(self.model, 'model'):
                    p = next(self.model.model.parameters())
                    if p.dtype == torch.float16:
                        is_half = True
                        self.model.model.float() # Cast to Float for CLIP
            except Exception as e:
                print(f"Warning checking dtype: {e}")
                
            try:
                self.model.set_classes(class_list)
            except Exception as e:
                print(f"Error setting classes (YOLO-World): {e}")
            
            # Restore Half precision if it was Half before
            if is_half:
                 try:
                    self.model.model.half()
                 except Exception as e:
                    print(f"Warning restoring half: {e}")
            
            self.filter_classes = None # World model handles filtering internally
            
        else:
            # Standard YOLO model (Fallback to index filtering)
            print("Standard YOLO model detected. Using index filtering.")
            indices = []
            # self.model.names is a dict {0: 'person', ...}
            # Invert it for lookup: {'person': 0, ...}
            name_to_id = {v: k for k, v in self.model.names.items()}
            
            for name in class_list:
                if name in name_to_id:
                    indices.append(name_to_id[name])
            
            if not indices and class_list:
                print(f"Warning: None of the requested classes {class_list} exist in this model.")
                # If no matches, we might want to show nothing, or everything?
                # Let's show nothing if specific classes were requested but none found.
                self.filter_classes = [] 
            else:
                self.filter_classes = indices

    def load_master_vocabulary(self):
        # A comprehensive list of common objects for "Open Vocabulary" detection
        self.master_vocab = [
            # Electronics
            'person', 'smartphone', 'laptop', 'mouse', 'keyboard', 'monitor', 'headphones', 'headset', 
            'smartwatch', 'tablet', 'camera', 'television', 'remote control', 'game controller',
            'cell phone', 'tv', # COCO synonyms
            
            # Personal Items
            'eyeglasses', 'sunglasses', 'backpack', 'handbag', 'suitcase', 'wallet', 'watch', 'keys',
            'credit card', 'book', 'pen', 'pencil', 'notebook',
            
            # Household
            'cup', 'mug', 'bottle', 'glass', 'plate', 'fork', 'knife', 'spoon', 'bowl',
            'chair', 'couch', 'bed', 'table', 'lamp', 'clock', 'vase', 'scissors', 'toothbrush',
            'dining table', 'potted plant', # COCO
            
            # Vehicles
            'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'airplane', 'train', 'boat',
            
            # Animals
            'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]
        print(f"Loading Master Vocabulary ({len(self.master_vocab)} classes)...")
        self.set_classes(self.master_vocab)

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'ALL':
            # For standard models, ALL means "Reset Filter" (None)
            # For World models, it means "Set Master Vocab"
            if hasattr(self.model, 'set_classes'):
                self.set_classes(self.master_vocab)
            else:
                self.filter_classes = None # Reset to all 80 COCO classes
                print("Mode: ALL (Standard Model - All Classes)")
                
        elif mode == 'LIVING':
            living = ['person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
            self.set_classes(living)
        elif mode == 'OBJECTS':
            # For objects, we exclude living things
            # This is tricky for standard models if we don't have a full list.
            # Easier to just list common objects.
            objects = [c for c in self.master_vocab if c not in ['person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']]
            self.set_classes(objects)
        return self.mode

    def get_color(self, id):
        if id not in self.colors:
            random.seed(id)
            self.colors[id] = [random.randint(50, 255) for _ in range(3)]
        return self.colors[id]

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        return self.tracking_enabled

    def detect(self, frame):
        # FP16 (half=True) for speed if CUDA
        # tracker="botsort.yaml" for better re-id
        use_half = (self.device == 'cuda')
        
        # Apply filter_classes if set (for standard models)
        classes_arg = self.filter_classes
        
        if self.tracking_enabled:
            results = self.model.track(frame, persist=True, verbose=False, device=self.device, half=use_half, tracker="botsort.yaml", classes=classes_arg)
        else:
            results = self.model(frame, verbose=False, device=self.device, half=use_half, classes=classes_arg)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls]
                
                # ID for tracking
                track_id = int(box.id[0]) if box.id is not None else cls
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': name,
                    'conf': conf,
                    'id': track_id
                })
        
        return detections

