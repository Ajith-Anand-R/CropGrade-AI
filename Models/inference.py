import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DETECTOR_WEIGHTS = BASE_DIR / "Models" / "best.pt"

# Model loader / caching
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        if DETECTOR_WEIGHTS.exists():
            print(f"Loading YOLOv8 detector from {DETECTOR_WEIGHTS}")
            _detector = YOLO(str(DETECTOR_WEIGHTS))
        else:
            print("YOLOv8 detector weights not found. Running in simulated detector mode.")
            _detector = None
    return _detector

def detect_coin_opencv(cv_img):
    """
    Detects a circular ₹10 coin using OpenCV Hough Circles.
    Returns:
        coin_diameter_px: Diameter of the coin in pixels, or None if not found.
        coin_circle: (x, y, r) tuple of the detected coin circle, or None.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    
    # Hough Circles parameter tuning
    # dp=1.2, minDist=100, param1=50 (Canny threshold), param2=35 (accumulator threshold)
    # minRadius=15, maxRadius=100 (expects the coin to be clearly visible)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=50, 
        param2=35, 
        minRadius=15, 
        maxRadius=120
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Select the circle that is most circular/largest or simply the first one
        # In tomato box context, the ₹10 coin is usually the most clean circle
        best_circle = circles[0] # [x, y, r]
        coin_diameter_px = int(best_circle[2] * 2)
        return coin_diameter_px, (int(best_circle[0]), int(best_circle[1]), int(best_circle[2]))
        
    return None, None

def calculate_iou(boxA, boxB):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def get_shelf_life(grade, ripeness, defect):
    """
    Heuristics to calculate estimated shelf life.
    """
    if defect != "none":
        if defect == "rot":
            return "1-2 days"
        else: # bruise or crack
            return "2-4 days"
            
    if grade == "grade_a":
        if ripeness == "green":
            return "10-14 days"
        elif ripeness == "pink":
            return "7-10 days"
        elif ripeness == "red":
            return "5-7 days"
        else: # overripe
            return "3-4 days"
    elif grade == "grade_b":
        if ripeness == "green":
            return "7-9 days"
        elif ripeness == "pink":
            return "5-7 days"
        elif ripeness == "red":
            return "3-5 days"
        else:
            return "2-3 days"
    else: # grade_c
        return "1-2 days"

def run_simulated_inference(cv_img):
    """
    Simulates tomato detections and OpenCV coin detection for testing and development.
    """
    h, w, _ = cv_img.shape
    
    # 1. Simulate ₹10 coin detection
    coin_x = int(w * 0.8)
    coin_y = int(h * 0.8)
    coin_r = int(w * 0.05) # 5% of width
    coin_diameter_px = coin_r * 2
    coin_circle = (coin_x, coin_y, coin_r)
    
    num_tomatoes = np.random.randint(4, 9)
    detections = []
    
    # Bbox pool
    for i in range(num_tomatoes):
        tx = int(w * (0.15 + 0.08 * i))
        ty = int(h * (0.3 + 0.05 * (i % 2)))
        # random physical sizes
        actual_size_mm = np.random.uniform(30, 75)
        
        # Calculate bounding box pixels based on actual size
        # actual_size = (tomato_pixels / coin_pixels) * 27 -> tomato_pixels = (actual_size / 27) * coin_pixels
        tomato_size_px = int((actual_size_mm / 27.0) * coin_diameter_px)
        
        x1 = tx
        y1 = ty
        x2 = x1 + tomato_size_px
        y2 = y1 + tomato_size_px
        
        # Categorize
        if actual_size_mm < 40:
            size_cat = "Small"
        elif actual_size_mm < 60:
            size_cat = "Medium"
        else:
            size_cat = "Large"
            
        grade = np.random.choice(["grade_a", "grade_b", "grade_c"], p=[0.5, 0.3, 0.2])
        ripeness = np.random.choice(["green", "pink", "red", "overripe"], p=[0.1, 0.3, 0.5, 0.1])
        defect = np.random.choice(["none", "none", "none", "crack", "bruise", "rot"], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
        
        # Force grade C if rotten
        if defect == "rot":
            grade = "grade_c"
            
        shelf_life = get_shelf_life(grade, ripeness, defect)
        
        detections.append({
            "id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "size_mm": round(actual_size_mm, 1),
            "size_category": size_cat,
            "ripeness": ripeness,
            "defect": defect,
            "grade": grade,
            "shelf_life": shelf_life
        })
        
    return detections, coin_circle

def predict_batch(image_path):
    """
    Loads image, detects ₹10 coin using OpenCV, detects tomatoes with YOLOv8, 
    and returns parsed quality, ripeness, defect, and size predictions.
    """
    cv_img = cv2.imread(str(image_path))
    if cv_img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    h, w, _ = cv_img.shape
    detector = get_detector()
    
    # 1. Run OpenCV ₹10 coin detection
    coin_diameter_px, coin_circle = detect_coin_opencv(cv_img)
    
    # 2. Check if we need to fall back to simulation
    if detector is None:
        detections, coin_circle = run_simulated_inference(cv_img)
    else:
        # Run YOLOv8 model
        results = detector(cv_img)
        
        # YOLO classes:
        # 0: grade_a, 1: grade_b, 2: grade_c, 
        # 3: green, 4: pink, 5: red, 6: overripe, 
        # 7: crack, 8: bruise, 9: rot
        raw_tomatoes = []
        raw_colors = []
        raw_defects = []
        
        for result in results:
            for box in result.boxes:
                coords = [int(c) for c in box.xyxy[0].cpu().numpy().tolist()]
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                if cls_id in [0, 1, 2]: # Tomato Grades
                    raw_tomatoes.append({"bbox": coords, "cls": cls_id, "conf": conf})
                elif cls_id in [3, 4, 5, 6]: # Ripeness colors
                    raw_colors.append({"bbox": coords, "cls": cls_id, "conf": conf})
                elif cls_id in [7, 8, 9]: # Defects
                    raw_defects.append({"bbox": coords, "cls": cls_id, "conf": conf})
                    
        # Group overlapping boxes using IoU
        detections = []
        class_names = {
            0: "grade_a", 1: "grade_b", 2: "grade_c",
            3: "green", 4: "pink", 5: "red", 6: "overripe",
            7: "crack", 8: "bruise", 9: "rot"
        }
        
        # If coin is not detected, use a default fallback diameter (10% of image width)
        scale_diameter_px = int(coin_diameter_px) if coin_diameter_px else int(w * 0.1)
        
        for idx, t in enumerate(raw_tomatoes):
            t_box = t["bbox"]
            grade = class_names[t["cls"]]
            
            # Find overlapping ripeness color
            best_color = "red" # default fallback
            max_color_iou = 0
            for c in raw_colors:
                iou = calculate_iou(t_box, c["bbox"])
                if iou > max_color_iou and iou > 0.2:
                    max_color_iou = iou
                    best_color = class_names[c["cls"]]
                    
            # Find overlapping defect (highest confidence)
            best_defect = "none"
            max_defect_iou = 0
            for d in raw_defects:
                iou = calculate_iou(t_box, d["bbox"])
                if iou > max_defect_iou and iou > 0.15:
                    max_defect_iou = iou
                    best_defect = class_names[d["cls"]]
            
            # Size estimation via OpenCV coin scale
            # tomato_pixels = max of width or height
            tomato_px = int(max(t_box[2] - t_box[0], t_box[3] - t_box[1]))
            actual_size_mm = float((tomato_px / scale_diameter_px) * 27.0)
            
            # Size Category
            if actual_size_mm < 40:
                size_cat = "Small"
            elif actual_size_mm < 60:
                size_cat = "Medium"
            else:
                size_cat = "Large"
                
            shelf_life = get_shelf_life(grade, best_color, best_defect)
            
            detections.append({
                "id": int(idx + 1),
                "bbox": [int(c) for c in t_box],
                "size_mm": float(round(actual_size_mm, 1)),
                "size_category": size_cat,
                "ripeness": best_color,
                "defect": best_defect,
                "grade": grade,
                "shelf_life": shelf_life
            })
            
    # Calculate batch summary metrics
    total_tomatoes = len(detections)
    if total_tomatoes > 0:
        grades = [d["grade"] for d in detections]
        a_count = grades.count("grade_a")
        b_count = grades.count("grade_b")
        c_count = grades.count("grade_c")
        
        ripeness_counts = {}
        for d in detections:
            ripeness_counts[d["ripeness"]] = ripeness_counts.get(d["ripeness"], 0) + 1
        dominant_ripeness = max(ripeness_counts, key=ripeness_counts.get)
        
        # Average Size
        sizes = [d["size_mm"] for d in detections]
        avg_size = int(round(sum(sizes) / len(sizes))) if sizes else 0

        # Overall Batch Grade & Market Info
        if c_count > total_tomatoes * 0.3:
            batch_grade = "Grade C"
            recommendation = "Reject or process immediately. High percentage of low-quality or damaged tomatoes."
            market_value = 12
            recommended_buyer = "Local Market"
        elif a_count > total_tomatoes * 0.7:
            batch_grade = "Grade A"
            recommendation = "Premium quality batch. High shelf-life. Excellent for export or premium retail."
            market_value = 32
            recommended_buyer = "Exporter"
        else:
            batch_grade = "Grade B"
            recommendation = "Standard market quality. Distribute locally. Shelf life is stable."
            market_value = 22
            recommended_buyer = "Retail Chain"

        # Shelf Life Days
        if dominant_ripeness == "green":
            shelf_life_days = 10
        elif dominant_ripeness == "pink":
            shelf_life_days = 7
        elif dominant_ripeness == "red":
            shelf_life_days = 5
        else:
            shelf_life_days = 2

        # If any tomato is rotten, cap shelf life at 1 day
        if any(d["defect"] == "rot" for d in detections):
            shelf_life_days = 1
    else:
        batch_grade = "N/A"
        a_count = b_count = c_count = 0
        dominant_ripeness = "N/A"
        recommendation = "No tomatoes detected. Position your camera directly over the tomato box."
        avg_size = 0
        shelf_life_days = 0
        market_value = 0
        recommended_buyer = "None"
        
    summary = {
        "total_count": total_tomatoes,
        "batch_grade": batch_grade,
        "grade_distribution": {
            "A": a_count,
            "B": b_count,
            "C": c_count
        },
        "dominant_ripeness": dominant_ripeness,
        "coin_detected": coin_circle is not None,
        "recommendation": recommendation,
        "avg_size": avg_size,
        "shelf_life_days": shelf_life_days,
        "market_value": market_value,
        "recommended_buyer": recommended_buyer
    }
    
    return {
        "summary": summary,
        "detections": detections,
        "coin_circle": coin_circle  # Pass coordinates to display on image
    }

if __name__ == "__main__":
    # Test script with dummy image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "Test Coin Circle", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Draw a solid circle to test circle detection
    cv2.circle(dummy_img, (320, 240), 40, (255, 255, 255), -1)
    
    test_path = BASE_DIR / "Models" / "temp_test_image.jpg"
    cv2.imwrite(str(test_path), dummy_img)
    
    try:
        results = predict_batch(test_path)
        import pprint
        pprint.pprint(results)
    finally:
        if test_path.exists():
            os.remove(test_path)
