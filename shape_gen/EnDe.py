import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt  # used internally; output is via Streamlit
from scipy.spatial import Delaunay
import random
from io import BytesIO

# ----- Provided Functions with Modifications -----

def generate_max_random_circles(image_size=(512, 512), min_radius=50, max_radius=100, 
                                max_attempts=50000, max_fail_attempts=10000, max_circles_limit=10):
    img = np.zeros(image_size, dtype=np.uint8)
    circles = []

    def is_too_close(x, y, radius):
        for (cx, cy, cr) in circles:
            distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            if distance < (cr + radius):
                return True
        return False

    attempts = 0
    failed_attempts = 0

    while (attempts < max_attempts and failed_attempts < max_fail_attempts and 
           len(circles) < max_circles_limit):
        used_space = sum([np.pi * cr**2 for (_, _, cr) in circles])
        total_space = image_size[0] * image_size[1]
        remaining_space = total_space - used_space

        remaining_capacity = remaining_space / total_space
        min_dynamic_radius = int(min_radius + (remaining_capacity * (max_radius - min_radius)))
        max_dynamic_radius = int(min_dynamic_radius * 1.5)
        max_dynamic_radius = min(max_dynamic_radius, max_radius)

        radius = random.randint(min_dynamic_radius, max_dynamic_radius)
        center_x = random.randint(radius, image_size[1] - radius)
        center_y = random.randint(radius, image_size[0] - radius)

        if not is_too_close(center_x, center_y, radius):
            cv2.circle(img, (center_x, center_y), radius, 255, 1)
            circles.append((center_x, center_y, radius))
            failed_attempts = 0
        else:
            failed_attempts += 1
        attempts += 1

    inverted_img = 255 - img
    return inverted_img, len(circles), circles

def resize_image_to_shape(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

def compute_average_under_circles(input_image, circle_image, circles):
    output_image = np.zeros((input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8)
    resized_circle_image = cv2.resize(circle_image, (input_image.shape[1], input_image.shape[0]))
    for (cx, cy, radius) in circles:
        y, x = np.ogrid[:resized_circle_image.shape[0], :resized_circle_image.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        circle_pixels = input_image[mask]
        if len(circle_pixels) > 0:
            average_value = np.mean(circle_pixels, axis=0)
        else:
            average_value = [0, 0, 0]
        output_image[mask] = np.append(average_value, [255])
    return output_image

def overlay_mask_on_image(input_image, mask_image):
    output_image = input_image.copy()
    mask_alpha = mask_image[:, :, 3] / 255.0
    for c in range(3):
        output_image[:, :, c] = (1 - mask_alpha) * input_image[:, :, c] + mask_alpha * mask_image[:, :, c]
    return output_image

def encode(input_image, shape_type, output_path, **kwargs):
    shape_type = shape_type.lower()
    if shape_type in ['triangle', 'triangles']:
        image_orig = input_image
        image_resized = cv2.resize(image_orig, (500, 500))
        padding = 30
        image_padded = cv2.copyMakeBorder(image_resized, padding, padding, padding, padding, cv2.BORDER_REFLECT)
        h_pad, w_pad, _ = image_padded.shape
        num_triangles = kwargs.get('num_triangles', 510)
        num_points = num_triangles + 2
        points = np.array([[random.randint(0, w_pad), random.randint(0, h_pad)] 
                           for _ in range(num_points)])
        tri = Delaunay(points)
        overlay_img = image_resized.copy()
        boundaries = []
        for simplex in tri.simplices:
            triangle_points = points[simplex]
            triangle_pts_no_padding = triangle_points - [padding, padding]
            boundaries.append(triangle_pts_no_padding)
            mask = np.zeros((h_pad, w_pad), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(triangle_points), 255)
            masked_image = cv2.bitwise_and(image_padded, image_padded, mask=mask)
            avg_color = cv2.mean(masked_image, mask=mask)[:3]
            avg_color_bgr = tuple(map(int, (avg_color[2], avg_color[1], avg_color[0])))
            overlay_temp = overlay_img.copy()
            cv2.fillConvexPoly(overlay_temp, np.int32(triangle_pts_no_padding), avg_color_bgr)
            alpha = 0.7
            overlay_img = cv2.addWeighted(overlay_img, 1 - alpha, overlay_temp, alpha, 0)
        original_resized = image_resized

    elif shape_type in ['rectangle', 'rectangles']:
        image_orig = input_image
        image_resized = cv2.resize(image_orig, (256, 256))
        h, w, _ = image_resized.shape
        img_mask = np.zeros((h, w), dtype=np.uint8)
        boundaries = []
        attempts = 0
        failed_attempts = 0
        max_attempts = kwargs.get('max_attempts', 500000)
        max_fail_attempts = kwargs.get('max_fail_attempts', 10000)
        max_rectangles_limit = kwargs.get('max_rectangles_limit', 653)
        min_size = kwargs.get('min_size', (5, 5))
        max_size = kwargs.get('max_size', (10, 10))
        while attempts < max_attempts and failed_attempts < max_fail_attempts and len(boundaries) < max_rectangles_limit:
            used_space = sum([rw * rh for (_, _, rw, rh) in boundaries])
            total_space = h * w
            remaining_capacity = (total_space - used_space) / total_space
            min_dynamic_width = int(min_size[0] + remaining_capacity * (max_size[0] - min_size[0]))
            min_dynamic_height = int(min_size[1] + remaining_capacity * (max_size[1] - min_size[1]))
            max_dynamic_width = min(int(min_dynamic_width * 1.5), max_size[0])
            max_dynamic_height = min(int(min_dynamic_height * 1.5), max_size[1])
            width = random.randint(min_dynamic_width, max_dynamic_width)
            height = random.randint(min_dynamic_height, max_dynamic_height)
            x = random.randint(0, w - width)
            y = random.randint(0, h - height)
            too_close = False
            for (rx, ry, rw, rh) in boundaries:
                if rx < x + width and rx + rw > x and ry < y + height and ry + rh > y:
                    too_close = True
                    break
            if not too_close:
                cv2.rectangle(img_mask, (x, y), (x + width, y + height), 255, 1)
                boundaries.append((x, y, width, height))
                failed_attempts = 0
            else:
                failed_attempts += 1
            attempts += 1
        inverted_mask = 255 - img_mask
        output_mask = np.zeros((h, w, 4), dtype=np.uint8)
        for (x, y, width, height) in boundaries:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y:y + height, x:x + width] = 255
            rect_pixels = image_resized[mask == 255]
            if len(rect_pixels) > 0:
                average_value = np.mean(rect_pixels, axis=0)
            else:
                average_value = [0, 0, 0]
            output_mask[mask == 255] = np.append(average_value, [255])
        overlay_img = image_resized.copy()
        mask_alpha = output_mask[:, :, 3] / 255.0
        for c in range(3):
            overlay_img[:, :, c] = (1 - mask_alpha) * image_resized[:, :, c] + mask_alpha * output_mask[:, :, c]
        original_resized = image_resized

    elif shape_type in ['circle', 'circles']:
        image_orig = input_image
        image_resized = cv2.resize(image_orig, (512, 512))
        h, w, _ = image_resized.shape
        circle_image, num_circles, circles = generate_max_random_circles(
            image_size=(h, w),
            max_circles_limit=kwargs.get('max_circles_limit', 1900),
            min_radius=kwargs.get('min_radius', 5),
            max_radius=kwargs.get('max_radius', 10),
            max_attempts=kwargs.get('max_attempts', 10000),
            max_fail_attempts=kwargs.get('max_fail_attempts', 10000)
        )
        resized_circle_image = resize_image_to_shape(circle_image, image_resized.shape[:2])
        averaged_circle_image = compute_average_under_circles(image_resized, resized_circle_image, circles)
        overlay_img = overlay_mask_on_image(image_resized, averaged_circle_image)
        original_resized = image_resized
        boundaries = circles

    else:
        raise ValueError("Unsupported shape type. Choose from 'triangles', 'rectangles', or 'circles'.")

    # Create an encoding mask for boundaries.
    encode_mask = np.zeros(original_resized.shape[:2], dtype=np.uint8)
    if shape_type in ['triangle', 'triangles']:
        for tri in boundaries:
            pts = np.int32(tri)
            cv2.polylines(encode_mask, [pts], isClosed=True, color=255, thickness=1)
    elif shape_type in ['rectangle', 'rectangles']:
        for (x, y, width, height) in boundaries:
            cv2.rectangle(encode_mask, (x, y), (x + width, y + height), 255, thickness=1)
    elif shape_type in ['circle', 'circles']:
        for (cx, cy, radius) in boundaries:
            cv2.circle(encode_mask, (cx, cy), radius, 255, thickness=1)

    encoded_image = overlay_img.copy()
    # Encode boundary information into the blue channel's LSB.
    for i in range(encoded_image.shape[0]):
        for j in range(encoded_image.shape[1]):
            if encode_mask[i, j] == 255:
                encoded_image[i, j] = original_resized[i, j]
                encoded_image[i, j, 0] = (encoded_image[i, j, 0] & 254) | 1
            else:
                encoded_image[i, j, 0] = encoded_image[i, j, 0] & 254

    # Add corner markers for validation (3x3 blocks in each corner)
    corner_size = 3
    h_img, w_img, _ = encoded_image.shape
    corner_positions = {
        "top_left": (0, 0),
        "top_right": (0, w_img - corner_size),
        "bottom_left": (h_img - corner_size, 0),
        "bottom_right": (h_img - corner_size, w_img - corner_size)
    }
    expected_patterns = {
        "top_left": (1, 1, 1),
        "top_right": (0, 0, 1),
        "bottom_left": (0, 1, 0),
        "bottom_right": (1, 0, 0)
    }
    for corner, (y, x) in corner_positions.items():
        exp_b, exp_g, exp_r = expected_patterns[corner]
        for i in range(y, y + corner_size):
            for j in range(x, x + corner_size):
                encoded_image[i, j, 0] = (encoded_image[i, j, 0] & 254) | exp_b
                encoded_image[i, j, 1] = (encoded_image[i, j, 1] & 254) | exp_g
                encoded_image[i, j, 2] = (encoded_image[i, j, 2] & 254) | exp_r

    if shape_type in ['circle', 'circles']:
        return encoded_image, boundaries
    else:
        return encoded_image, boundaries

def decode(encoded_image, shape_type, boundaries=None):
    shape_type = shape_type.lower()
    if encoded_image is None:
        st.error("Error: Encoded image is None.")
        return None, None, None
    h, w, _ = encoded_image.shape
    blue_lsb = encoded_image[:, :, 0] & 1
    corner_size = 3
    corner_positions = {
        "top_left": (0, 0),
        "top_right": (0, w - corner_size),
        "bottom_left": (h - corner_size, 0),
        "bottom_right": (h - corner_size, w - corner_size)
    }
    expected_patterns = {
        "top_left": (1, 1, 1),
        "top_right": (0, 0, 1),
        "bottom_left": (0, 1, 0),
        "bottom_right": (1, 0, 0)
    }
    threshold = 6
    valid = True
    for corner, (y, x) in corner_positions.items():
        exp_b, exp_g, exp_r = expected_patterns[corner]
        count_b = np.sum(blue_lsb[y:y+corner_size, x:x+corner_size] == exp_b)
        if count_b < threshold:
            valid = False
            st.warning(f"Corner '{corner}' failed validation.")
            break
    if valid:
        st.info("Valid encoding detected. Decoding boundaries.")
        binary_image = (blue_lsb * 255).astype(np.uint8)
    else:
        st.warning("No valid encoding found. Returning black binary image.")
        binary_image = np.zeros_like(blue_lsb, dtype=np.uint8)
    if cv2.countNonZero(binary_image) < 50:
        st.info("Binary image nearly empty; applying dilation.")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    rgb_values = []
    annotated = encoded_image.copy()
    if shape_type in ['triangle', 'triangles']:
        triangles = []
        if boundaries is not None:
            triangles = boundaries
        else:
            ret, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 3:
                    triangles.append(approx.reshape(-1, 2))
        for tri in triangles:
            pts = np.int32(tri)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            center = np.mean(tri, axis=0)
            center_x = int(np.clip(center[0], 0, w - 1))
            center_y = int(np.clip(center[1], 0, h - 1))
            b, g, r = encoded_image[center_y, center_x]
            rgb_values.append([r, g, b])
    elif shape_type in ['rectangle', 'rectangles']:
        ret, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            if w_rect > 1 and h_rect > 1:
                cv2.rectangle(annotated, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 1)
                center_x = x + w_rect // 2
                center_y = y + h_rect // 2
                b, g, r = encoded_image[center_y, center_x]
                rgb_values.append([r, g, b])
    elif shape_type in ['circle', 'circles']:
        ret, thresh = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 4 and radius < 12:
                cv2.circle(annotated, center, radius, (0, 255, 0), 1)
                b, g, r = encoded_image[center[1], center[0]]
                rgb_values.append([r, g, b])
    else:
        st.error("Unsupported shape type for decoding.")
    
    return binary_image, annotated, rgb_values
