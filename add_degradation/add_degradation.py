import cv2
import numpy as np
import random


def motion_blur(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    degree = max(5, int(intensity * 30))
    angle = random.uniform(0, 360)
    
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel /= np.sum(kernel)
    
    return cv2.filter2D(img, -1, kernel)


def lens_blur(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    kernel_size = int(3 + intensity * 300) | 1
    sigma = intensity * 20
    
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    
    blurred = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        blurred[..., c] = cv2.filter2D(img[..., c].astype(np.float32), -1, kernel)
    
    result = cv2.addWeighted(
        img, 1 - intensity * 0.7,
        blurred.astype(np.uint8), intensity * 0.9, 0
    )
    
    return result


def gaussian_noise(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    noise_std = intensity * 75
    noise = np.random.normal(0, noise_std, img.shape)
    result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return result


def block_exchange(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    h, w = img.shape[:2]
    block_size = min(32, int(5 + intensity * 30))
    noisy_img = img.copy()
    
    num_exchanges = int(intensity * 35)
    for _ in range(num_exchanges):
        i1 = random.randint(0, h // block_size - 1)
        j1 = random.randint(0, w // block_size - 1)
        i2 = random.randint(0, h // block_size - 1)
        j2 = random.randint(0, w // block_size - 1)
        
        y1, x1 = i1 * block_size, j1 * block_size
        y2, x2 = i2 * block_size, j2 * block_size
        
        block1 = noisy_img[y1:y1+block_size, x1:x1+block_size].copy()
        noisy_img[y1:y1+block_size, x1:x1+block_size] = \
            noisy_img[y2:y2+block_size, x2:x2+block_size]
        noisy_img[y2:y2+block_size, x2:x2+block_size] = block1
    
    return noisy_img


def jpeg_compression(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be in range [0.0, 1.0]")
    
    if img is None:
        raise ValueError("Input image is None")
    
    quality = int(100 - intensity * 95)
    quality = max(5, min(100, quality))
    
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_params)
    compressed_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    return compressed_img


def mean_shift(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    spatial_radius = int(intensity * 40)
    color_radius = int(intensity * 40)
    
    return cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius)


def color_diffusion(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    kernel_size = 3 + 2 * int(intensity * 20)
    sigma = intensity * 50
    
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T * (intensity ** 2)
    
    diffused = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        diffused[..., c] = cv2.filter2D(img[..., c].astype(np.float32), -1, kernel)
    
    if intensity > 0.9:
        h, w = img.shape[:2]
        for _ in range(int(100 * intensity)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(5, 20)
            cv2.circle(diffused, (x, y), radius,
                      (np.random.randint(0, 255),) * 3, -1)
    
    result = cv2.addWeighted(
        img, max(0.1, 1 - intensity * 0.9),
        diffused.astype(np.uint8), min(0.9, intensity * 0.9), 0
    )
    
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpness_change(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    if intensity > 0:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * (intensity * 80)
        result = cv2.filter2D(img, -1, kernel)
    else:
        ksize = int(3 + abs(intensity) * 5) | 1
        result = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    result = cv2.addWeighted(img, 0.7, result, 0.3, 0)
    return result


def dark_illumination(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    result = (img * (1 - intensity ** 2)).clip(0, 255).astype(np.uint8)
    return result


def hsv_saturation(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= (1 - intensity)
    result = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result


def atmospheric_turbulence(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distortion = intensity * 40 * np.sin(y / 30 + intensity * 5)
    x_new = np.clip(x + distortion, 0, w - 1).astype(np.float32)
    y_new = np.clip(y + distortion * 0.7, 0, h - 1).astype(np.float32)
    
    return cv2.remap(img, x_new, y_new, cv2.INTER_LINEAR)


def dirty_lens(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    h, w = img.shape[:2]
    dirt = np.zeros((h, w, 3), dtype=np.float32)
    
    if intensity > 0.1:
        for _ in range(int(10 * intensity)):
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            cv2.ellipse(dirt, (center_x, center_y),
                       (random.randint(150, 300), random.randint(100, 200)),
                       angle=random.randint(0, 180),
                       startAngle=0, endAngle=360,
                       color=(50, 50, 50), thickness=-1)
    
    for _ in range(int(300 * intensity)):
        x = random.randint(0, w)
        y = random.randint(0, h)
        cv2.circle(dirt, (x, y), random.randint(4, 20),
                  (random.randint(50, 100),) * 3, -1)
    
    if intensity > 0.5:
        for _ in range(int(5 * intensity)):
            x = random.randint(0, w)
            y = random.randint(0, h)
            cv2.circle(dirt, (x, y), random.randint(20, 50),
                      (80, 80, 80), -1)
            cv2.circle(dirt, (x, y), random.randint(10, 30),
                      (120, 120, 120), -1)
    
    dirt = cv2.GaussianBlur(dirt, (0, 0), 30)
    dirt = dirt.astype(np.uint8)
    
    result = cv2.addWeighted(img, 1 - 0.7 * intensity, dirt, 0.8 * intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def scan_lines(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    line_interval = max(3, int(20 / (intensity + 0.1)))
    line_width = max(5, int(7 * intensity))
    
    result = img.copy()
    for i in range(0, img.shape[0], line_interval):
        end_line = min(i + line_width, img.shape[0])
        result[i:end_line] = result[i:end_line] * 0.01
    
    return result


def graffiti(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be in range [0.0, 1.0]")
    
    if img is None:
        raise ValueError("Input image is None")
    
    h, w = img.shape[:2]
    result = img.copy()
    
    for _ in range(int(10 * intensity)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pt1 = (random.randint(0, w - 1), random.randint(0, h - 1))
        pt2 = (random.randint(0, w - 1), random.randint(0, h - 1))
        thickness = random.randint(1, max(1, int(5 * intensity)))
        cv2.line(result, pt1, pt2, color, thickness)
    
    if intensity > 0.55:
        texts = ["X", "FAKE", "COPY", "VOID", "COPYRIGHT", str(random.randint(1, 100))]
        text = random.choice(texts)
        
        font_scale = max(0.5, intensity * 5)
        thickness = max(1, int(font_scale))
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_width, text_height = text_size
        
        if h - 10 > text_height + 10:
            text_x = random.randint(0, max(1, w - text_width - 10))
            text_y = random.randint(text_height + 10, h - 10)
            
            cv2.putText(result, text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       (0, 0, 255),
                       thickness)
    
    return result


def watermark_damage(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(int(1 + intensity * 15)):
        x = random.randint(0, w - 50)
        y = random.randint(0, h - 50)
        cv2.rectangle(mask, (x, y),
                     (x + random.randint(50, 200), y + random.randint(20, 80)), 1, -1)
    
    repaired = cv2.inpaint(img, (mask * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    result = cv2.addWeighted(img, 1 - intensity, repaired, intensity, 0)
    
    if intensity > 0.5:
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        result[edges > 0] = result[edges > 0] * 0.8
    
    return result


def lens_flare(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    
    h, w = img.shape[:2]
    flare = np.zeros((h, w, 3), dtype=np.float32)
    
    num_flares = 3 + int(30 * intensity)
    for _ in range(num_flares):
        x = random.randint(0, w)
        y = random.randint(0, h)
        radius = random.randint(10, 50)
        color = np.array([255, 255, 235])
        
        cv2.circle(flare, (x, y), radius, color.tolist(), -1)
        
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(30, 150)
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        cv2.line(flare, (x, y), (end_x, end_y), color.tolist(), 2)
    
    flare = cv2.GaussianBlur(flare, (3, 3), 20 * intensity)
    
    result = cv2.addWeighted(img.astype(np.float32), 1, flare, 0.9 * intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)
