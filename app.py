import cv2
import os
import numpy as np
import math
import time

INPUT_FOLDER = 'static/input'
OUTPUT_FOLDER = 'static/output'
VIDEO_OUTPUT_FOLDER = 'static/video_output'

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not load image: {image_path}")
        return None

    # Resize to standard size
    img = cv2.resize(img, (960, 540))
    height, width = img.shape[:2]
    
    # Create debug image to show processing steps
    debug_img = img.copy()
    
    print(f"[*] Processing image: {os.path.basename(image_path)}")
    
    # Step 1: Advanced color filtering
    color_mask = advanced_color_detection(img)
    
    # Step 2: Edge detection
    edges = enhanced_edge_detection(img)
    
    # Step 3: Combine masks
    combined = cv2.bitwise_or(edges, color_mask)
    
    # Step 4: Apply region of interest
    roi_mask = create_dynamic_roi(combined, width, height)
    masked_edges = cv2.bitwise_and(combined, roi_mask)
    
    # Step 5: Detect lanes using multiple methods
    lane_lines = detect_lane_lines(masked_edges, width, height)
    
    # Step 6: Draw results
    result = draw_lane_detection(img, lane_lines, width, height)
    
    return result

def advanced_color_detection(img):
    """Enhanced color detection for various lighting conditions"""
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # White lane detection (multiple methods)
    # Method 1: HSV white detection
    lower_white_hsv = np.array([0, 0, 180])
    upper_white_hsv = np.array([180, 30, 255])
    white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    
    # Method 2: LAB white detection
    lower_white_lab = np.array([120, 0, 0])
    upper_white_lab = np.array([255, 130, 130])
    white_mask_lab = cv2.inRange(lab, lower_white_lab, upper_white_lab)
    
    # Method 3: Grayscale threshold for bright areas
    _, white_mask_gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Yellow lane detection
    lower_yellow = np.array([10, 50, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine all masks
    white_mask = cv2.bitwise_or(white_mask_hsv, white_mask_lab)
    white_mask = cv2.bitwise_or(white_mask, white_mask_gray)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    return color_mask

def enhanced_edge_detection(img):
    """Multi-scale edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple Gaussian blurs
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Multiple Canny edge detection with different parameters
    edges1 = cv2.Canny(blur1, 30, 90)
    edges2 = cv2.Canny(blur2, 50, 120)
    edges3 = cv2.Canny(blur1, 20, 60)
    
    # Combine edge detections
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)
    
    return edges

def create_dynamic_roi(image, width, height):
    """Create a more flexible ROI that adapts to image content"""
    mask = np.zeros_like(image)
    
    # Main trapezoid ROI
    roi_vertices = np.array([[
        (int(width * 0.02), height),
        (int(width * 0.42), int(height * 0.52)),
        (int(width * 0.58), int(height * 0.52)),
        (int(width * 0.98), height)
    ]], np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Add upper region for distant lanes
    upper_roi = np.array([[
        (int(width * 0.35), int(height * 0.52)),
        (int(width * 0.46), int(height * 0.35)),
        (int(width * 0.54), int(height * 0.35)),
        (int(width * 0.65), int(height * 0.52))
    ]], np.int32)
    
    cv2.fillPoly(mask, upper_roi, 255)
    
    return mask

def detect_lane_lines(masked_edges, width, height):
    """Detect lane lines using multiple methods"""
    all_lines = []
    
    # Method 1: Standard Hough Lines
    lines1 = cv2.HoughLinesP(
        masked_edges, rho=1, theta=np.pi/180, threshold=20,
        minLineLength=30, maxLineGap=80
    )
    if lines1 is not None:
        all_lines.extend(lines1)
    
    # Method 2: More sensitive detection
    lines2 = cv2.HoughLinesP(
        masked_edges, rho=1, theta=np.pi/180, threshold=15,
        minLineLength=25, maxLineGap=120
    )
    if lines2 is not None:
        all_lines.extend(lines2)
    
    # Method 3: Detect shorter segments for curves
    lines3 = cv2.HoughLinesP(
        masked_edges, rho=2, theta=np.pi/90, threshold=10,
        minLineLength=20, maxLineGap=150
    )
    if lines3 is not None:
        all_lines.extend(lines3)
    
    if not all_lines:
        print("[!] No lines detected")
        return {"left": [], "right": [], "center": []}
    
    print(f"[*] Detected {len(all_lines)} line segments")
    
    # Filter and classify lines
    classified_lines = classify_lines(all_lines, width, height)
    
    # Process each group
    processed_lines = {}
    for side in ['left', 'right', 'center']:
        if classified_lines[side]:
            processed_lines[side] = process_line_group(classified_lines[side], height)
        else:
            processed_lines[side] = []
    
    return processed_lines

def classify_lines(lines, width, height):
    """Classify lines into left, right, or center lanes"""
    left_lines = []
    right_lines = []
    center_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line properties
        if x2 - x1 == 0:
            slope = float('inf')
        else:
            slope = (y2 - y1) / (x2 - x1)
        
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        center_x = (x1 + x2) / 2
        
        # Skip very short lines
        if length < 15:
            continue
        
        # Classify based on slope and position
        if abs(slope) < 0.2:  # Nearly horizontal lines
            continue
        
        # Left lane detection
        if slope < -0.2 and center_x < width * 0.7:
            left_lines.append(line[0])
        # Right lane detection  
        elif slope > 0.2 and center_x > width * 0.3:
            right_lines.append(line[0])
        # Center lines (less strict slope requirements)
        elif abs(slope) > 0.1 and width * 0.3 < center_x < width * 0.7:
            center_lines.append(line[0])
    
    print(f"[*] Classified - Left: {len(left_lines)}, Right: {len(right_lines)}, Center: {len(center_lines)}")
    
    return {"left": left_lines, "right": right_lines, "center": center_lines}

def process_line_group(lines, height):
    """Process a group of lines to create smooth lane representation"""
    if not lines:
        return []
    
    # Sort lines by y-coordinate (top to bottom)
    lines_sorted = sorted(lines, key=lambda line: min(line[1], line[3]))
    
    # For curves, keep multiple segments
    if len(lines_sorted) <= 3:
        return lines_sorted
    
    # Group nearby lines
    grouped_lines = []
    current_group = [lines_sorted[0]]
    
    for i in range(1, len(lines_sorted)):
        current_line = lines_sorted[i]
        last_line = current_group[-1]
        
        # Check if lines are close enough to group
        y_dist = abs(min(current_line[1], current_line[3]) - min(last_line[1], last_line[3]))
        
        if y_dist < height * 0.2:  # Within 20% of image height
            current_group.append(current_line)
        else:
            # Average current group and start new group
            if len(current_group) > 1:
                avg_line = average_line_group(current_group)
                if avg_line:
                    grouped_lines.append(avg_line)
            else:
                grouped_lines.extend(current_group)
            current_group = [current_line]
    
    # Don't forget the last group
    if len(current_group) > 1:
        avg_line = average_line_group(current_group)
        if avg_line:
            grouped_lines.append(avg_line)
    else:
        grouped_lines.extend(current_group)
    
    return grouped_lines

def average_line_group(lines):
    """Average a group of similar lines"""
    if not lines:
        return None
    
    x1_avg = sum(line[0] for line in lines) // len(lines)
    y1_avg = sum(line[1] for line in lines) // len(lines)
    x2_avg = sum(line[2] for line in lines) // len(lines)
    y2_avg = sum(line[3] for line in lines) // len(lines)
    
    return [x1_avg, y1_avg, x2_avg, y2_avg]

def draw_lane_detection(img, lane_lines, width, height):
    """Draw detected lanes with different colors and styles"""
    result = img.copy()
    
    # Colors for different lane types
    colors = {
        'left': (0, 255, 0),      # Green
        'right': (0, 255, 0),     # Green  
        'center': (0, 255, 255)   # Yellow
    }
    
    line_count = 0
    
    # Draw each type of lane
    for lane_type, lines in lane_lines.items():
        color = colors[lane_type]
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Draw main line
            cv2.line(result, (x1, y1), (x2, y2), color, 6)
            
            # Draw endpoints for visibility
            cv2.circle(result, (x1, y1), 4, color, -1)
            cv2.circle(result, (x2, y2), 4, color, -1)
            
            line_count += 1
    
    # Draw lane fill if we have both left and right lanes
    if lane_lines['left'] and lane_lines['right']:
        result = draw_lane_fill(result, lane_lines['left'], lane_lines['right'], width, height)
    
    # Add text overlay with detection info
    cv2.putText(result, f"Lines detected: {line_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    lane_types = [k for k, v in lane_lines.items() if v]
    cv2.putText(result, f"Lane types: {', '.join(lane_types)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print(f"[*] Drew {line_count} lane lines")
    
    return result

def draw_lane_fill(img, left_lines, right_lines, width, height):
    """Fill the area between left and right lanes"""
    overlay = img.copy()
    
    # Get the longest lines from each side
    left_line = max(left_lines, key=lambda line: 
                   math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2))
    right_line = max(right_lines, key=lambda line: 
                    math.sqrt((line[2]-line[0])**2 + (line[3]-line[1])**2))
    
    # Extend lines to image boundaries
    left_extended = extend_line(left_line, height)
    right_extended = extend_line(right_line, height)
    
    if left_extended and right_extended:
        # Create polygon for lane area
        polygon = np.array([
            [left_extended[0], left_extended[1]],
            [left_extended[2], left_extended[3]], 
            [right_extended[2], right_extended[3]],
            [right_extended[0], right_extended[1]]
        ], np.int32)
        
        cv2.fillPoly(overlay, [polygon], (0, 255, 255))
        
        # Blend with original image
        img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
    
    return img

def extend_line(line, height):
    """Extend a line to the bottom of the image"""
    x1, y1, x2, y2 = line
    
    if y2 == y1:  # Horizontal line
        return None
    
    # Calculate slope
    slope = (x2 - x1) / (y2 - y1)
    
    # Extend to bottom
    bottom_x = int(x1 + slope * (height - y1))
    
    # Extend to top of line segment
    top_y = min(y1, y2)
    if top_y == y1:
        top_x = x1
    else:
        top_x = x2
    
    return [bottom_x, height, top_x, top_y]

def process_video(video_path, output_path):
    """Process a video file for lane detection"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[!] Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[*] Video info - FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (960, 540))
    
    if not out.isOpened():
        print(f"[!] Could not create output video: {output_path}")
        cap.release()
        return False
    
    frame_count = 0
    start_time = time.time()
    
    # Lane tracking variables for temporal consistency
    previous_lanes = {"left": [], "right": [], "center": []}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame (reuse image processing function)
        processed_frame = process_frame_for_video(frame, previous_lanes)
        
        if processed_frame is not None:
            out.write(processed_frame)
        else:
            # If processing fails, write original frame
            resized_frame = cv2.resize(frame, (960, 540))
            out.write(resized_frame)
        
        frame_count += 1
        
        # Progress update every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            print(f"[*] Progress: {progress:.1f}% ({frame_count}/{total_frames}) - ETA: {eta:.1f}s")
    
    # Clean up
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    print(f"[✓] Video processing completed in {processing_time:.1f} seconds")
    print(f"[✓] Output saved: {output_path}")
    
    return True

def process_frame_for_video(frame, previous_lanes):
    """Process a single video frame with temporal consistency"""
    if frame is None:
        return None
    
    # Resize to standard size
    frame = cv2.resize(frame, (960, 540))
    height, width = frame.shape[:2]
    
    # Step 1: Advanced color filtering
    color_mask = advanced_color_detection(frame)
    
    # Step 2: Edge detection
    edges = enhanced_edge_detection(frame)
    
    # Step 3: Combine masks
    combined = cv2.bitwise_or(edges, color_mask)
    
    # Step 4: Apply region of interest
    roi_mask = create_dynamic_roi(combined, width, height)
    masked_edges = cv2.bitwise_and(combined, roi_mask)
    
    # Step 5: Detect lanes
    lane_lines = detect_lane_lines(masked_edges, width, height)
    
    # Step 6: Apply temporal smoothing
    smoothed_lanes = apply_temporal_smoothing(lane_lines, previous_lanes)
    
    # Step 7: Draw results
    result = draw_lane_detection(frame, smoothed_lanes, width, height)
    
    # Update previous lanes for next frame
    previous_lanes.update(smoothed_lanes)
    
    return result

def apply_temporal_smoothing(current_lanes, previous_lanes, alpha=0.3):
    """Apply temporal smoothing to reduce lane flickering in videos"""
    smoothed_lanes = {"left": [], "right": [], "center": []}
    
    for lane_type in ['left', 'right', 'center']:
        current = current_lanes.get(lane_type, [])
        previous = previous_lanes.get(lane_type, [])
        
        if not current and not previous:
            continue
        
        if not current and previous:
            # Use previous lanes with reduced confidence
            smoothed_lanes[lane_type] = previous[:1]  # Keep only best previous lane
        elif current and not previous:
            # Use current lanes
            smoothed_lanes[lane_type] = current
        else:
            # Blend current and previous lanes
            smoothed_lines = []
            
            for i, curr_line in enumerate(current):
                if i < len(previous):
                    prev_line = previous[i]
                    # Weighted average of current and previous
                    smoothed_line = [
                        int(alpha * curr_line[0] + (1 - alpha) * prev_line[0]),
                        int(alpha * curr_line[1] + (1 - alpha) * prev_line[1]),
                        int(alpha * curr_line[2] + (1 - alpha) * prev_line[2]),
                        int(alpha * curr_line[3] + (1 - alpha) * prev_line[3])
                    ]
                    smoothed_lines.append(smoothed_line)
                else:
                    smoothed_lines.append(curr_line)
            
            smoothed_lanes[lane_type] = smoothed_lines
    
    return smoothed_lanes
def run_lane_detection():
    """Process all images and videos in the input folder"""
    if not os.path.exists(INPUT_FOLDER):
        print(f"[!] Input folder '{INPUT_FOLDER}' does not exist!")
        return
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
    
    # Get all files
    all_files = os.listdir(INPUT_FOLDER)
    
    # Separate images and videos
    image_files = [f for f in all_files 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    video_files = [f for f in all_files 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'))]
    
    print(f"[*] Found {len(image_files)} images and {len(video_files)} videos to process")
    
    # Process images
    if image_files:
        print(f"\n{'='*50}")
        print("PROCESSING IMAGES")
        print(f"{'='*50}")
        
        processed_images = 0
        for filename in image_files:
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            print(f"\n[*] Processing image: {filename}")
            processed = process_image(input_path)
            
            if processed is not None:
                success = cv2.imwrite(output_path, processed)
                if success:
                    print(f"[✓] Saved: {filename}")
                    processed_images += 1
                else:
                    print(f"[!] Failed to save: {filename}")
            else:
                print(f"[!] Failed to process: {filename}")
        
        print(f"\n[✓] Successfully processed {processed_images}/{len(image_files)} images")
    
    # Process videos
    if video_files:
        print(f"\n{'='*50}")
        print("PROCESSING VIDEOS")
        print(f"{'='*50}")
        
        processed_videos = 0
        for filename in video_files:
            input_path = os.path.join(INPUT_FOLDER, filename)
            
            # Create output filename with _processed suffix
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_processed.mp4"
            output_path = os.path.join(VIDEO_OUTPUT_FOLDER, output_filename)
            
            print(f"\n[*] Processing video: {filename}")
            success = process_video(input_path, output_path)
            
            if success:
                processed_videos += 1
            else:
                print(f"[!] Failed to process video: {filename}")
        
        print(f"\n[✓] Successfully processed {processed_videos}/{len(video_files)} videos")
    
    # Summary
    if not image_files and not video_files:
        print(f"[!] No supported files found in '{INPUT_FOLDER}'")
        print("[*] Supported formats:")
        print("    Images: PNG, JPG, JPEG, BMP, TIFF")
        print("    Videos: MP4, AVI, MOV, MKV, FLV, WMV, M4V")
    else:
        print(f"\n{'='*50}")
        print("PROCESSING COMPLETE")
        print(f"{'='*50}")
        if image_files:
            print(f"[*] Image results saved in: {OUTPUT_FOLDER}")
        if video_files:
            print(f"[*] Video results saved in: {VIDEO_OUTPUT_FOLDER}")

def batch_process_folder(input_folder, output_folder, file_types=['images', 'videos']):
    """Process a specific folder with custom settings"""
    global INPUT_FOLDER, OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER
    
    # Temporarily change paths
    original_input = INPUT_FOLDER
    original_output = OUTPUT_FOLDER
    original_video_output = VIDEO_OUTPUT_FOLDER
    
    INPUT_FOLDER = input_folder
    OUTPUT_FOLDER = output_folder
    VIDEO_OUTPUT_FOLDER = os.path.join(output_folder, 'videos')
    
    try:
        if 'images' in file_types and 'videos' in file_types:
            run_lane_detection()
        elif 'images' in file_types:
            # Process only images
            process_images_only()
        elif 'videos' in file_types:
            # Process only videos
            process_videos_only()
    finally:
        # Restore original paths
        INPUT_FOLDER = original_input
        OUTPUT_FOLDER = original_output
        VIDEO_OUTPUT_FOLDER = original_video_output

def process_images_only():
    """Process only image files"""
    if not os.path.exists(INPUT_FOLDER):
        print(f"[!] Input folder '{INPUT_FOLDER}' does not exist!")
        return
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    image_files = [f for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"[!] No image files found in '{INPUT_FOLDER}'")
        return
    
    print(f"[*] Processing {len(image_files)} images...")
    
    processed_count = 0
    for filename in image_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        processed = process_image(input_path)
        if processed is not None:
            if cv2.imwrite(output_path, processed):
                processed_count += 1
                print(f"[✓] {filename}")
            else:
                print(f"[!] Failed to save: {filename}")
        else:
            print(f"[!] Failed to process: {filename}")
    
    print(f"[✓] Processed {processed_count}/{len(image_files)} images")

def process_videos_only():
    """Process only video files"""
    if not os.path.exists(INPUT_FOLDER):
        print(f"[!] Input folder '{INPUT_FOLDER}' does not exist!")
        return
    
    os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
    
    video_files = [f for f in os.listdir(INPUT_FOLDER) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'))]
    
    if not video_files:
        print(f"[!] No video files found in '{INPUT_FOLDER}'")
        return
    
    print(f"[*] Processing {len(video_files)} videos...")
    
    processed_count = 0
    for filename in video_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_processed.mp4"
        output_path = os.path.join(VIDEO_OUTPUT_FOLDER, output_filename)
        
        if process_video(input_path, output_path):
            processed_count += 1
    
    print(f"[✓] Processed {processed_count}/{len(video_files)} videos")

if __name__ == '__main__':
    print("Lane Detection System")
    print("=" * 50)
    print("Supports: Images (PNG, JPG, JPEG, BMP, TIFF)")
    print("         Videos (MP4, AVI, MOV, MKV, FLV, WMV, M4V)")
    print("=" * 50)
    
    run_lane_detection()