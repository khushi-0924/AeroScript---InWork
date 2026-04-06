import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands 

# Initialize video capture from default camera (0)
cap = cv2.VideoCapture(0)

# ===== CANVAS AND DRAWING VARIABLES =====
# canvas: stores all drawn content as numpy array
# prev_x, prev_y: tracks previous cursor position for smooth line drawing
canvas = None
prev_x, prev_y = None, None

# ===== DRAWING MODE SETTINGS =====
# draw_color: current brush color in BGR format (Red, Green, Blue)
# brush_thickness: size of the brush circle in pixels
draw_color = (0, 0, 200)  # Default: Red
brush_thickness = 10

# ===== MODE FLAGS =====
# white_mode: toggle between camera background and white background
# eraser_mode: toggle between drawing and erasing
white_mode = False
eraser_mode = False

# ===== PINCH GESTURE VARIABLES =====
# Used for canvas panning (moving the drawing when pinching thumb and index)
# prev_pinch_x, prev_pinch_y: tracks pinch position for smooth canvas movement
# pinch_active: flag to detect if user is currently pinching
prev_pinch_x, prev_pinch_y = None, None
pinch_active = False

# ===== MAIN HAND DETECTION LOOP =====
# Creates a hand detection model with confidence thresholds
with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    # Continuous loop: read frames from camera until exit
    while True:
        # Capture frame from camera
        success, frame = cap.read()
        if not success:
            break

        # Flip frame horizontally so it acts like a mirror (more intuitive for users)
        frame = cv2.flip(frame, 1)

        # Initialize canvas on first frame with same dimensions as video
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Create white canvas option for white background mode
        white_canvas = np.ones_like(frame) * 255

        # Convert frame from BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame to detect hand landmarks and get results
        results = hands.process(rgb_frame)

        # Get frame dimensions for coordinate calculations
        h, w, _ = frame.shape

        # ===== UI SETTINGS =====
        # Divide screen width into 5 sections for color buttons and other controls
        color_w = w // 5  # Each button takes 1/5 of screen width
        bar_h = 80  # Height of control bar at top

        # ===== BIN SETTINGS =====
        # Delete button appears at bottom center when user is pinching
        # Used to clear the canvas when pinch gesture is dropped into bin
        bin_w, bin_h = 100, 60
        bin_x1, bin_y1 = w//2 - bin_w//2, h - bin_h
        bin_x2, bin_y2 = bin_x1 + bin_w, h

        # Initialize cursor coordinates for hand pointer
        cx, cy = None, None

        # ===== HAND DETECTION AND PROCESSING =====
        # If hands are detected in the frame
        if results.multi_hand_landmarks:
            # Process each detected hand (should be only 1 due to max_num_hands=1)
            for hand_landmarks in results.multi_hand_landmarks:

                # Extract key landmarks from hand:
                # Landmark 8: Index finger tip (drawing cursor)
                # Landmark 4: Thumb tip (used for pinch detection)
                # Landmark 6: Index finger PIP (middle joint, for finger up detection)
                # Landmark 12: Middle finger tip
                # Landmark 10: Middle finger PIP
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                index_pip = hand_landmarks.landmark[6]
                middle_tip = hand_landmarks.landmark[12]
                middle_pip = hand_landmarks.landmark[10]

                # Convert normalized coordinates (0-1) to pixel coordinates (0-width/height)
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # ===== FINGER STATE DETECTION =====
                # Check if fingers are pointing up (y-coordinate of tip < pip means finger is up)
                # This determines whether user is in drawing mode or selection mode
                index_up = index_tip.y < index_pip.y
                middle_up = middle_tip.y < middle_pip.y

                # ===== PINCH DETECTION =====
                # Calculate distance between thumb tip and index finger tip
                # If distance < 40 pixels, user is pinching (gesture activated)
                # Pinch is used for panning the canvas around
                dist = np.hypot(cx - tx, cy - ty)

                if dist < 40:
                    # Pinch detected: pan the canvas
                    pinch_active = True
                    alpha = 0.2  # Smoothing factor for stroke reduction (lower = smoother)

                    # Apply smoothing and move canvas
                    if prev_pinch_x is not None and prev_pinch_y is not None:
                        # Exponential moving average for smooth movement
                        smooth_x = int(alpha * cx + (1 - alpha) * prev_pinch_x)
                        smooth_y = int(alpha * cy + (1 - alpha) * prev_pinch_y)

                        # Calculate movement delta
                        dx = smooth_x - prev_pinch_x
                        dy = smooth_y - prev_pinch_y

                        # Roll (shift) canvas pixels in x and y directions
                        canvas = np.roll(canvas, shift=dx, axis=1)
                        canvas = np.roll(canvas, shift=dy, axis=0)

                        prev_pinch_x, prev_pinch_y = smooth_x, smooth_y
                    else:
                        # Initialize pinch starting position
                        prev_pinch_x, prev_pinch_y = cx, cy

                else:
                    # Pinch released: reset pinch state
                    pinch_active = False
                    prev_pinch_x, prev_pinch_y = None, None

                    # ===== SELECTION MODE =====
                    # When both index and middle fingers are up, user is in selection mode
                    # They can click UI buttons at the top to change colors or modes
                    # Reset drawing state when entering selection mode
                    prev_x, prev_y = None, None

                    # Check if cursor is in the top button bar area
                    if cy < bar_h:

                        # Left button: Select RED color
                        if 0 < cx < color_w:
                            draw_color = (0, 0, 200)
                            eraser_mode = False

                        # Second button: Select GREEN color
                        elif color_w < cx < 2*color_w:
                            draw_color = (0, 200, 0)
                            eraser_mode = False

                        # Third button: Select BLUE color
                        elif 2*color_w < cx < 3*color_w:
                            draw_color = (200, 0, 0)
                            eraser_mode = False

                        # Fourth button: ERASER mode
                        elif 3*color_w < cx < 4*color_w:
                            eraser_mode = True

                        # Fifth button: Toggle WHITE BACKGROUND mode
                        elif 4*color_w < cx < 5*color_w:
                            white_mode = not white_mode
                            prev_x, prev_y = None, None

                    # ===== DRAW MODE =====
                    # When only index finger is up (middle finger down), user can draw
                    elif index_up:

                        # If we have a previous position, draw a line from there to current position
                        if prev_x is not None and prev_y is not None:
                            # Calculate distance between current and previous point
                            dx = cx - prev_x
                            dy = cy - prev_y
                            dist = int(np.hypot(dx, dy))
                            if dist == 0:
                                dist = 1

                            # Interpolate points between previous and current position
                            # This creates smooth curved lines instead of jagged segments
                            for i in range(dist):
                                x = int(prev_x + dx * i / dist)
                                y = int(prev_y + dy * i / dist)

                                # Draw or erase based on current mode
                                if eraser_mode:
                                    # Eraser: paint black circles to clear the canvas
                                    cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                                else:
                                    # Draw: paint colored circles with current brush color
                                    cv2.circle(canvas, (x, y), brush_thickness, draw_color, -1)

                        # Update position for next frame
                        prev_x, prev_y = cx, cy

                    # ===== NO DRAWING =====
                    # When neither finger is specifically positioned for drawing/selection
                    else:
                        prev_x, prev_y = None, None

        # ===== NO HAND DETECTED =====
        # If no hand is detected in the frame, reset drawing position
        else:
            prev_x, prev_y = None, None

        # ===== DELETE IF DROPPED IN BIN =====
        # If user is currently pinching and moves to bin location, clear the canvas
        # This allows users to delete their drawing by "dropping" the pinch into the bin
        if pinch_active and cx is not None and cy is not None:
            if bin_x1 < cx < bin_x2 and bin_y1 < cy < bin_y2:
                canvas = np.zeros_like(frame)

        # ===== BACKGROUND =====
        # Select background: white background if white_mode is True, otherwise use camera feed
        bg = white_canvas if white_mode else frame

        # ===== CANVAS AND BACKGROUND MERGING =====
        # Convert canvas to grayscale to create a mask
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        # Create binary mask: pixels with intensity > 20 are foreground (drawn content)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Invert mask to get background mask (where nothing is drawn)
        mask_inv = cv2.bitwise_not(mask)

        # Combine background and canvas:
        # - Use background where canvas is empty (mask_inv)
        # - Use canvas drawing where it exists (mask)
        combined = cv2.add(
            cv2.bitwise_and(bg, bg, mask=mask_inv),
            cv2.bitwise_and(canvas, canvas, mask=mask)
        )

        # ===== UI BUTTON BAR RENDERING =====
        # Draw 5 color/mode selector buttons at the top of the screen

        # Button 1: RED color selector (left section)
        cv2.rectangle(combined, (0, 0), (color_w, bar_h), (0, 0, 200), -1)
        
        # Button 2: GREEN color selector
        cv2.rectangle(combined, (color_w, 0), (2*color_w, bar_h), (0, 200, 0), -1)
        
        # Button 3: BLUE color selector
        cv2.rectangle(combined, (2*color_w, 0), (3*color_w, bar_h), (200, 0, 0), -1)

        # Button 4: ERASER mode (dark gray background with white circle icon)
        cv2.rectangle(combined, (3*color_w, 0), (4*color_w, bar_h), (50, 50, 50), -1)
        cv2.circle(combined, (3*color_w + color_w//2, bar_h//2), 25, (255,255,255), -1)

        # Button 5: TOGGLE WHITE BACKGROUND (dark gray with white rectangle outline)
        cv2.rectangle(combined, (4*color_w, 0), (5*color_w, bar_h), (80, 80, 80), -1)
        cv2.rectangle(combined,
                      (4*color_w+20, 20),
                      (5*color_w-20, bar_h-20),
                      (255,255,255), 3)

        # ===== BIN DISPLAY (ONLY DURING PINCH) =====
        # Show delete bin at bottom center only when user is actively pinching
        # User can drag pinch gesture into bin to clear the entire canvas
        if pinch_active:
            cv2.rectangle(combined, (bin_x1, bin_y1), (bin_x2, bin_y2), (0,0,255), -1)
            cv2.putText(combined, "BIN", (bin_x1+20, bin_y1+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # ===== HAND POINTER INDICATOR =====
        # Draw cursor on screen showing current index finger tip position
        # Black outer circle + cyan inner circle for visibility
        if cx is not None and cy is not None:
            cv2.circle(combined, (cx, cy), 10, (0,0,0), -1)
            cv2.circle(combined, (cx, cy), 6, (0,255,255), -1)

        # ===== DISPLAY ON SCREEN =====
        # Show the combined image (background + canvas + UI) in a window
        cv2.imshow("Hand Painter", combined)

        # ===== KEYBOARD INPUT HANDLING =====
        # Wait 1ms for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key (27): Exit the application
        if key == 27:
            break
        
        # 'c' key: Clear the canvas (shortcut to delete all drawings)
        if key == ord('c'):
            canvas = np.zeros_like(frame)

# ===== CLEANUP =====
# Close camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
