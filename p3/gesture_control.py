# gesture_control.py
import cv2
import mediapipe as mp
import serial
import time
import numpy as np
from threading import Thread, Lock
import json

class GestureController:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Serial communication
        self.ser = serial.Serial('COM7', 9600, timeout=1)
        self.current_command = "S"
        self.command_history = []
        
        # Path tracking
        self.path_position = 0
        self.path_coordinates = [(0, 0)]  # (x, y) coordinates
        
        # Thread control
        self.lock = Lock()
        self.running = True
        self.frame = None
        
        # Start processing thread
        self.thread = Thread(target=self.process_gestures)
        self.thread.daemon = True
        self.thread.start()
    
    def process_gestures(self):
        while self.running:
            success, img = self.cap.read()
            if not success:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            fingers = {"4": 0, "8": 0, "12": 0, "16": 0, "20": 0}
            numberOfFingers = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    landmarks = []
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append((id, cx, cy))
                    
                    if landmarks:
                        # Thumb (different for left/right hand)
                        handedness = results.multi_handedness[0].classification[0].label
                        if handedness == 'Left':
                            if landmarks[4][1] > landmarks[3][1]:  # Thumb left
                                fingers["4"] = 1
                        else:
                            if landmarks[4][1] < landmarks[3][1]:  # Thumb right
                                fingers["4"] = 1
                                
                        # Other fingers
                        if landmarks[8][2] < landmarks[6][2]:  # Index
                            fingers["8"] = 1
                        if landmarks[12][2] < landmarks[10][2]:  # Middle
                            fingers["12"] = 1
                        if landmarks[16][2] < landmarks[14][2]:  # Ring
                            fingers["16"] = 1
                        if landmarks[20][2] < landmarks[18][2]:  # Pinky
                            fingers["20"] = 1
            
            numberOfFingers = sum(fingers.values())
            
            # Map gesture to command
            if numberOfFingers == 1:
                command = "F"
            elif numberOfFingers == 2:
                command = "B"
            elif numberOfFingers == 3:
                command = "L"
            elif numberOfFingers == 4:
                command = "R"
            else:
                command = "S"
                
            if command != self.current_command:
                self.ser.write(command.encode())
                self.current_command = command
                self.update_path(command)
                self.command_history.append(command)
                if len(self.command_history) > 10:
                    self.command_history.pop(0)
            
            # Add UI elements
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'Fingers: {numberOfFingers}', (10, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f'Fingers: {numberOfFingers}', (10, 30), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f'Command: {self.current_command}', (10, 60), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f'Command: {self.current_command}', (10, 60), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Update frame with lock
            with self.lock:
                ret, buffer = cv2.imencode('.jpg', img)
                self.frame = buffer.tobytes()
    
    def update_path(self, command):
        """Update the path coordinates based on the current command"""
        x, y = self.path_coordinates[-1]
        
        if command == "F":
            y -= 10
        elif command == "B":
            y += 10
        elif command == "L":
            x -= 10
        elif command == "R":
            x += 10
        
        self.path_coordinates.append((x, y))
        if len(self.path_coordinates) > 50:  # Limit history
            self.path_coordinates.pop(0)
    
    def get_path_data(self):
        """Return path data for visualization"""
        # Normalize coordinates for display
        if not self.path_coordinates:
            return []
            
        min_x = min(p[0] for p in self.path_coordinates)
        max_x = max(p[0] for p in self.path_coordinates)
        min_y = min(p[1] for p in self.path_coordinates)
        max_y = max(p[1] for p in self.path_coordinates)
        
        # Avoid division by zero
        x_range = max(1, max_x - min_x)
        y_range = max(1, max_y - min_y)
        
        normalized = []
        for x, y in self.path_coordinates:
            nx = (x - min_x) / x_range * 100
            ny = (y - min_y) / y_range * 100
            normalized.append((nx, ny))
            
        return normalized
    
    def get_frame(self):
        with self.lock:
            return self.frame
    
    def get_status(self):
        """Return current status as JSON"""
        return {
            "command": self.current_command,
            "path": self.get_path_data(),
            "history": self.command_history[-5:]  # Last 5 commands
        }
    
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        self.ser.close()     