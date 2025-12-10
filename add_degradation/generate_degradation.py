import add_degradation
import cv2
import os
import numpy as np
import argparse

DEGRADATION_CONFIG = {
    'capture': {
        'lens_blur': {'weight': 20},
        'lens_flare': {'weight': 20},
        'motion_blur': {'weight': 20},
        'dirty_lens': {'weight': 20},
        'hsv_saturation': {'weight': 20}
    },
    'transmission': {
        'jpeg_compression': {'weight': 25},
        'block_exchange': {'weight': 25},
        'mean_shift': {'weight': 25},
        'scan_lines': {'weight': 25}
    },
    'environment': {
        'dark_illumination': {'weight': 25},
        'atmospheric_turbulence': {'weight': 25},
        'gaussian_noise': {'weight': 25},
        'color_diffusion': {'weight': 25}
    },
    'postprocessing': {
        'sharpness_change': {'weight': 33},
        'graffiti': {'weight': 33},
        'watermark_damage': {'weight': 34}
    }
}

def apply_degradation_Benchmark(image, method_name, intensity):
    degradation_func = getattr(add_degradation, method_name)
    degraded_img = degradation_func(image, intensity)
    return degraded_img

def main():
    parser = argparse.ArgumentParser(description='Image degradation pipeline for robustness evaluation')
    parser.add_argument('--input_dir', type=str, 
                       default=os.getenv('INPUT_DIR', './data/images'),
                       help='Input image directory path (can be set via INPUT_DIR environment variable)')
    parser.add_argument('--output_base_dir', type=str,
                       default=os.getenv('OUTPUT_BASE_DIR', './data/output'),
                       help='Base directory for output images (can be set via OUTPUT_BASE_DIR environment variable)')
    parser.add_argument('--dataset_name', type=str,
                       default=os.getenv('DATASET_NAME', 'RealWorldQA'),
                       help='Dataset name (used to generate output directory names)')
    
    args = parser.parse_args()
    
    folder_path = args.input_dir
    output_base_dir = args.output_base_dir
    dataset_name = args.dataset_name
    
    output_dirs = {
        0.9: os.path.join(output_base_dir, f'{dataset_name}_Robust_100'),
        0.45: os.path.join(output_base_dir, f'{dataset_name}_Robust_50'),
        0.23: os.path.join(output_base_dir, f'{dataset_name}_Robust_25')
    }

    if not os.path.exists(folder_path):
        raise ValueError(f"Input directory does not exist: {folder_path}")
    
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    all_methods_with_weights = []
    for category, methods in DEGRADATION_CONFIG.items():
        for method_name, details in methods.items():
            all_methods_with_weights.append((method_name, details['weight']))

    method_names = [item[0] for item in all_methods_with_weights]
    weights = [item[1] for item in all_methods_with_weights]

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    num = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipping")
                num += 1
                continue

            selected_method_name = np.random.choice(method_names, p=probabilities)

            for intensity, output_dir in output_dirs.items():
                degraded_img = apply_degradation_Benchmark(image, selected_method_name, intensity)
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, degraded_img)
            
            num += 1
            if num % 100 == 0:
                print(f"Processed {num} images")
        
    print("Processing completed!")

if __name__ == '__main__':
    main()
