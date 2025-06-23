#!/usr/bin/env python3
"""
Standalone script to generate holographic encoding enhanced figure
for the Resonant Field Theory paper.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import traceback
import time
from datetime import datetime
import warnings
from PIL import Image, ImageDraw, ImageFont
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check for MLX
try:
    import mlx.core as mx
    logger.info("MLX successfully imported for GPU acceleration")
    HAS_MLX = True
except ImportError:
    logger.warning("MLX not available. Falling back to CPU processing.")
    HAS_MLX = False

try:
    from pdf2image import convert_from_path
except ImportError as e:
    logger.error(f"Required package missing: {e}")
    logger.error("Please install required packages with: pip install pdf2image pillow numpy matplotlib opencv-python")
    sys.exit(1)

# Use existing enhanced figures directory
OUTPUT_DIR = "/Users/okok/crystalineconciousnessai/Resonant Field Theory/figures_enhanced_20250503_161506"

def pdf_to_numpy(pdf_path):
    """Convert PDF to numpy array using pdf2image"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        logger.info(f"Converting PDF to image: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=300)
        # Use first page of the PDF
        pil_image = images[0]
        # Convert PIL Image to numpy array
        return np.array(pil_image)
    except Exception as e:
        logger.error(f"Error converting PDF to image: {e}")
        raise

def numpy_to_pdf(np_image, output_path):
    """Convert numpy array to PDF"""
    try:
        logger.info(f"Saving image to PDF: {output_path}")
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        pil_image = Image.fromarray(np_image.astype(np.uint8))
        temp_path = output_path.replace('.pdf', '.png')
        pil_image.save(temp_path, dpi=(300, 300))
        
        # Use matplotlib to convert PNG to PDF (preserves quality)
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.imshow(np_image)
        plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return output_path
    except Exception as e:
        logger.error(f"Error saving image to PDF: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise

def apply_mlx_gpu_processing(img_array):
    """Apply GPU-accelerated processing using MLX"""
    try:
        logger.info("Using MLX for GPU-accelerated processing")
        # Convert to MLX array for GPU processing
        mx_array = mx.array(img_array.astype(np.float32))
        
        # Normalize to [0, 1]
        mx_array = (mx_array - mx.min(mx_array)) / (mx.max(mx_array) - mx.min(mx_array))
        
        # Apply contrast enhancement
        mx_array = mx.power(mx_array, 1.5)
        
        # Apply additional sharpening using convolution
        kernel = mx.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=mx.float32)
        kernel = kernel.reshape(3, 3, 1, 1)
        
        # Reshape for conv operation
        mx_reshaped = mx.transpose(mx_array, (2, 0, 1))  # [C, H, W]
        mx_reshaped = mx_reshaped.reshape(1, 3, mx_reshaped.shape[1], mx_reshaped.shape[2])
        
        # Apply convolution for each channel
        sharpened_channels = []
        for c in range(3):
            channel = mx_reshaped[:, c:c+1, :, :]
            try:
                # In MLX 0.25, padding must be an integer or tuple
                sharpened = mx.conv2d(channel, kernel, padding=1)
                sharpened_channels.append(sharpened)
            except TypeError:
                # Fallback if padding='same' not supported
                sharpened = mx.conv2d(channel, kernel)
                sharpened_channels.append(sharpened)
        
        # Combine channels and reshape back
        mx_sharp = mx.concatenate(sharpened_channels, axis=1)
        mx_sharp = mx_sharp[0]  # Remove batch dimension
        mx_sharp = mx.transpose(mx_sharp, (1, 2, 0))  # [H, W, C]
        
        # Clip values to valid range
        mx_sharp = mx.clip(mx_sharp, 0, 1)
        
        # Convert back to numpy and scale to [0, 255]
        result = (mx_sharp * 255).astype(np.uint8).numpy()
        
        return result
    except Exception as e:
        logger.error(f"Error in GPU processing: {e}")
        logger.error(traceback.format_exc())
        # Fallback to CPU processing
        return apply_cpu_processing(img_array)

def apply_cpu_processing(img_array):
    """CPU-based image processing fallback"""
    logger.info("Using CPU processing (MLX not available)")
    
    # Convert to float32
    img_float = img_array.astype(np.float32)
    
    # Normalize to [0, 1]
    img_norm = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float))
    
    # Apply contrast enhancement
    img_contrast = np.power(img_norm, 1.5)
    
    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = np.zeros_like(img_contrast)
    
    # Apply to each channel
    for c in range(3):
        sharpened[:,:,c] = cv2.filter2D(img_contrast[:,:,c], -1, kernel)
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 1)
    
    # Scale to [0, 255]
    result = (sharpened * 255).astype(np.uint8)
    
    return result

def apply_phase_coherence(img):
    """Create phase coherence visualization"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        magnitude = np.abs(f_transform)
        phase = np.angle(f_transform)
        
        # Normalize phase to [0, 255] for visualization
        phase_normalized = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        
        # Apply colormap
        colored_phase = cv2.applyColorMap(phase_normalized, cv2.COLORMAP_PLASMA)
        
        # Apply edge detection to highlight boundaries
        edges = cv2.Canny(phase_normalized, 50, 150)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Combine phase colors with edge highlights
        alpha = 0.7
        result = cv2.addWeighted(colored_phase, alpha, edges_rgb, 0.3, 0)
        
        # Add contour lines at specific phase values
        contours = []
        for threshold in np.linspace(0, 255, 8):
            _, contour = cv2.threshold(phase_normalized, threshold, 255, cv2.THRESH_BINARY)
            contours.append(contour)
        
        # Add contour lines to the result
        for contour in contours:
            contour_edges = cv2.Canny(contour, 50, 150)
            contour_rgb = cv2.cvtColor(contour_edges, cv2.COLOR_GRAY2RGB)
            # Make contour lines bright white
            contour_rgb[contour_edges > 0] = [255, 255, 255]
            # Add to result
            result = cv2.addWeighted(result, 0.9, contour_rgb, 0.1, 0)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in phase coherence processing: {e}")
        logger.error(traceback.format_exc())
        # Just return the original image if processing fails
        return img

def apply_interference_pattern(img):
    """Create interference pattern visualization"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # Normalize magnitude to [0, 255]
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply colormap
        colormap = cv2.applyColorMap(magnitude_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)
        
        # Add directional arrows to show energy flow
        step = img.shape[1] // 20  # Number of arrows to display
        threshold = np.percentile(magnitude, 70)  # Only show arrows for top 30% of magnitudes
        
        for y in range(step, img.shape[0], step):
            for x in range(step, img.shape[1], step):
                if magnitude[y, x] > threshold:
                    # Get direction angle
                    angle = direction[y, x]
                    # Arrow length based on magnitude
                    length = int(20 * magnitude[y, x] / np.max(magnitude))
                    # Calculate end point
                    end_x = int(x + length * np.cos(angle))
                    end_y = int(y + length * np.sin(angle))
                    # Ensure within bounds
                    end_x = min(max(end_x, 0), img.shape[1]-1)
                    end_y = min(max(end_y, 0), img.shape[0]-1)
                    # Draw arrow
                    cv2.arrowedLine(colormap, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.3)
        
        return colormap
        
    except Exception as e:
        logger.error(f"Error in interference pattern processing: {e}")
        logger.error(traceback.format_exc())
        # Just return the original image if processing fails
        return img

def apply_multiscale_analysis(img):
    """Create multi-scale analysis visualization"""
    try:
        # Create a copy for processing
        result = img.copy()
        h, w = result.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw multi-scale geometric guides
        scales = [1.0, 0.75, 0.5, 0.25]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for scale, color in zip(scales, colors):
            size = int(min(w, h) * scale)
            
            # Draw square
            x1, y1 = center_x - size//2, center_y - size//2
            x2, y2 = center_x + size//2, center_y + size//2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw diagonal guides
            cv2.line(result, (x1, y1), (x2, y2), color, 1)
            cv2.line(result, (x2, y1), (x1, y2), color, 1)
            
            # Draw circles at corners
            radius = 5
            for corner in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
                cv2.circle(result, corner, radius, color, -1)
                
        # Add magnified regions of interest
        regions = [
            ((w//4, h//4), "Self-Similarity"),
            ((3*w//4, h//4), "Scale Transition"),
            ((w//4, 3*h//4), "Phase Structure"),
            ((3*w//4, 3*h//4), "Information Flow")
        ]
        
        for (x, y), label in regions:
            # Extract and magnify region
            size = 50
            x1, y1 = max(0, x-size), max(0, y-size)
            x2, y2 = min(w, x+size), min(h, y+size)
            
            if x2 > x1 and y2 > y1:  # Check if region is valid
                region = result[y1:y2, x1:x2]
                if region.size > 0:
                    magnified = cv2.resize(region, (size*2, size*2))
                    
                    # Add border
                    cv2.rectangle(magnified, (0, 0), (size*2-1, size*2-1), 
                                (255, 255, 255), 2)
                    
                    # Add label
                    cv2.putText(magnified, label, (10, 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 1)
                    
                    # Insert magnified region
                    y_insert = y - size*2 if y > h//2 else y + size*2
                    x_insert = x - size*2 if x > w//2 else x + size*2
                    
                    y_insert = max(0, min(h - size*2, y_insert))
                    x_insert = max(0, min(w - size*2, x_insert))
                    
                    result[y_insert:y_insert + size*2, 
                          x_insert:x_insert + size*2] = magnified
                        
                    # Draw connection line
                    cv2.line(result, (x, y), 
                            (x_insert + size, y_insert + size),
                            (255, 255, 0), 1)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in multi-scale analysis: {e}")
        logger.error(traceback.format_exc())
        return img

def enhance_holographic_encoding(pdf_path):
    """
    Enhance holographic encoding visualization (Figure 9) with:
    - Phase coherence transition mapping
    - Interference pattern analysis
    - Multi-scale self-similarity visualization
    - Information distribution guides
    """
    logger.info(f"Processing holographic encoding: {pdf_path}")
    
    try:
        # Convert PDF to numpy array
        img_array = pdf_to_numpy(pdf_path)
        
        # Create four-panel layout
        h, w, c = img_array.shape
        final = np.zeros((h*2, w*2, c), dtype=np.uint8)
        
        # Panel 1: Original with basic enhancement (top-left)
        if HAS_MLX:
            enhanced = apply_mlx_gpu_processing(img_array)
        else:
            enhanced = apply_cpu_processing(img_array)
        final[:h, :w] = enhanced
        
        # Panel 2: Phase coherence visualization (top-right)
        phase_panel = apply_phase_coherence(enhanced)
        final[:h, w:] = phase_panel
        
        # Panel 3: Interference pattern analysis (bottom-left)
        interference_panel = apply_interference_pattern(enhanced)
        final[h:, :w] = interference_panel
        
        # Panel 4: Multi-scale analysis (bottom-right)
        multiscale_panel = apply_multiscale_analysis(enhanced)
        final[h:, w:] = multiscale_panel
        
        # Add labels
        final_pil = Image.fromarray(final)
        draw = ImageDraw.Draw(final_pil)
        
        try:
            font = ImageFont.truetype("Arial", 30)
        except IOError:
            font = ImageFont.load_default()
            
        labels = [
            ("Original Pattern", (10, 10)),
            ("Phase Coherence Transitions", (w+10, 10)),
            ("Interference Pattern Dynamics", (10, h+10)),
            ("Multi-Scale Self-Similarity", (w+10, h+10))
        ]
        
        for text, pos in labels:
            bbox = draw.textbbox(pos, text, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill="black")
            draw.text(pos, text, fill="white", font=font)
            
        final = np.array(final_pil)
        
        # Save to existing enhanced figures directory
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        numpy_to_pdf(final, output_path)
        
        logger.info(f"Enhanced holographic encoding saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error enhancing holographic encoding: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Generate enhanced holographic encoding figure."""
    try:
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        holographic_path = os.path.join(figures_dir, "holographic_encoding.pdf")
        
        if not os.path.exists(holographic_path):
            raise FileNotFoundError(f"Could not find {holographic_path}")
            
        if not os.path.exists(OUTPUT_DIR):
            raise FileNotFoundError(f"Enhanced figures directory not found: {OUTPUT_DIR}")
            
        enhanced_path = enhance_holographic_encoding(holographic_path)
        logger.info(f"Successfully enhanced holographic encoding: {enhanced_path}")
        
        # Try to open the output directory
        if sys.platform == "darwin":  # macOS
            os.system(f"open {OUTPUT_DIR}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

