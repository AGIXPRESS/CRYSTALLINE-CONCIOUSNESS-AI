                    
                    # Add border to inset
                    magnified_draw = ImageDraw.Draw(magnified_pil)
                    magnified_draw.rectangle((0, 0, inset_width*2-1, inset_height*2-1), 
                                           outline=(255, 255, 0), width=3)
                    
                    # Convert back to numpy
                    magnified = np.array(magnified_pil)
                    
                    # Position the inset in a corner that doesn't overlap important content
                    # (simplified placement logic)
                    if x_center < width/2 and y_center < height/2:
                        inset_pos = (width - magnified.shape[1] - 10, 10)  # top-right
                    elif x_center >= width/2 and y_center < height/2:
                        inset_pos = (10, 10)  # top-left
                    elif x_center < width/2 and y_center >= height/2:
                        inset_pos = (width - magnified.shape[1] - 10, height - magnified.shape[0] - 10)  # bottom-right
                    else:
                        inset_pos = (10, height - magnified.shape[0] - 10)  # bottom-left
                    
                    # Paste the magnified inset onto the main image
                    pil_img_array = np.array(pil_img)
                    
                    # Draw a line connecting the inset to its source region
                    pil_img = Image.fromarray(pil_img_array)
                    draw = ImageDraw.Draw(pil_img)
                    draw.line((x_center, y_center, inset_pos[0] + magnified.shape[1]//2, 
                             inset_pos[1] + magnified.shape[0]//2), fill=(255, 255, 0), width=2)
                    
                    # Mark the source region
                    draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 255, 0), width=2)
                    
                    # Insert magnified region
                    pil_img_array = np.array(pil_img)
                    y_insert, x_insert = inset_pos
                    h_inset, w_inset = magnified.shape[:2]
                    
                    # Ensure inset fits within image boundaries
                    if (y_insert + h_inset <= height and x_insert + w_inset <= width):
                        pil_img_array[y_insert:y_insert + h_inset, x_insert:x_insert + w_inset] = magnified
                    
                    pil_img = Image.fromarray(pil_img_array)
                    draw = ImageDraw.Draw(pil_img)
        
        # Add color gradient to indicate evolution direction
        gradient_height = height // 15
        gradient = np.zeros((gradient_height, width, 3), dtype=np.uint8)
        
        # Create the gradient
        progress = i / (len(images) - 1) if len(images) > 1 else 0
        for x in range(width):
            rel_pos = x / width
            # Color based on position and progress
            if rel_pos < progress:
                # Already evolved (green to yellow)
                r = int(255 * rel_pos / progress) if progress > 0 else 0
                g = 255
                b = 0
            else:
                # Not yet evolved (blue to purple)
                r = int(255 * (rel_pos - progress) / (1 - progress)) if progress < 1 else 0
                g = 0
                b = 255
            
            gradient[:, x] = [r, g, b]
        
        # Add gradient to the bottom of the image
        pil_img_array = np.array(pil_img)
        pil_img_array[height - gradient_height:, :] = gradient
        
        # Add frame labels
        pil_img = Image.fromarray(pil_img_array)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("Arial", 30)
        except IOError:
            font = ImageFont.load_default()
        
        # Label showing progression
        stage_names = ["Initial State", "Mid Evolution", "Final State"]
        if i < len(stage_names):
            label = stage_names[i]
        else:
            label = f"Stage {i+1}"
        
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)
        draw.text((10, height - gradient_height - 40), 
                 f"Evolution Progress: {progress:.0%}", fill=(255, 255, 255), font=font)
        
        # Convert back to numpy array for saving
        enhanced = np.array(pil_img)
        
        # Save individual enhanced image
        base_filename = os.path.basename(path)
        output_path = os.path.join(OUTPUT_DIR, base_filename.replace('.pdf', '_enhanced.pdf'))
        numpy_to_pdf(enhanced, output_path)
        final_visualizations.append(output_path)
        
        logger.info(f"Enhanced {base_filename} saved to {output_path}")
    
        # Create a combined visualization showing all stages side by side
        # This is useful for direct comparison in the paper
        if len(enhanced_images) > 1:
            # Get dimensions
            h, w, c = enhanced_images[0].shape
        
        # Create a combined image with all stages
        combined_width = w * len(enhanced_images)
        combined = np.zeros((h, combined_width, c), dtype=np.uint8)
        
        # Place each image side by side
        for i, img in enumerate(enhanced_images):
            # Create enhanced version with boundaries
            enhanced = apply_mlx_gpu_processing(img)
            
            # Add label
            enhanced_pil = Image.fromarray(enhanced)
            draw = ImageDraw.Draw(enhanced_pil)
            
            try:
                font = ImageFont.truetype("Arial", 24)
            except IOError:
                font = ImageFont.load_default()
                
            if i == 0:
                label = "Initial State"
            elif i == len(enhanced_images) - 1:
                label = "Final State"
            else:
                label = f"Mid Evolution {i}"
                
            # Add white background for text visibility
            text_width, text_height = draw.textsize(label, font=font) if hasattr(draw, 'textsize') else (150, 30)
            draw.rectangle((10, 10, 10 + text_width + 10, 10 + text_height + 10), fill=(0, 0, 0))
            draw.text((15, 15), label, fill=(255, 255, 255), font=font)
            
            # Add difference highlights
            if i > 0:
                # Calculate difference from previous frame
                diff = cv2.absdiff(enhanced_images[i-1], enhanced_images[i])
                diff_mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) > 30
                
                # Highlight the differences
                enhanced_array = np.array(enhanced_pil)
                for c in range(3):
                    channel = enhanced_array[:,:,c]
                    if c == 0:  # Red channel
                        channel[diff_mask] = 255  # Highlight changes in red
                    else:
                        channel[diff_mask] = channel[diff_mask] // 2  # Dim other channels
                
                enhanced_pil = Image.fromarray(enhanced_array)
            
            # Convert back to numpy
            enhanced = np.array(enhanced_pil)
            
            # Add dividing line
            if i > 0:
                enhanced[:,0:3,:] = [255, 255, 0]  # Yellow line
            
            # Add to combined image
            combined[:, i*w:(i+1)*w] = enhanced
        
        # Add legend for the combined visualization
        legend_height = 50
        legend = np.ones((legend_height, combined_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Convert to PIL for text
        legend_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(legend_pil)
        
        try:
            font = ImageFont.truetype("Arial", 18)
        except IOError:
            font = ImageFont.load_default()
        
        # Add legend entries
        legend_entries = [
            ("Red", (255, 0, 0), "Areas of Change"),
            ("Cyan", (0, 255, 255), "Flow Direction"),
            ("Yellow", (255, 255, 0), "Magnified Region"),
            ("Magenta", (255, 0, 255), "Phase-Space Trajectory")
        ]
        
        # Position legend entries evenly
        entry_width = combined_width // len(legend_entries)
        for i, (color_name, color_rgb, description) in enumerate(legend_entries):
            # Draw color sample
            x_pos = i * entry_width + 10
            draw.rectangle((x_pos, 10, x_pos + 20, 30), fill=color_rgb, outline=(0, 0, 0))
            # Add text
            draw.text((x_pos + 25, 10), f"{color_name}: {description}", fill=(0, 0, 0), font=font)
        
        # Convert back to numpy
        legend = np.array(legend_pil)
        
        # Add legend to bottom of combined image
        combined_with_legend = np.vstack((combined, legend))
        
        # Save the combined visualization
        combined_output_path = os.path.join(OUTPUT_DIR, "field_evolution_combined_enhanced.pdf")
        numpy_to_pdf(combined_with_legend, combined_output_path)
        logger.info(f"Combined field evolution visualization saved to {combined_output_path}")
        
        # Create a color legend image for individual figures as well
        legend_width = width
        legend_height = 100
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add legend entries
        legend_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(legend_pil)
        
        # Add title
        draw.text((10, 10), "Visual Element Guide", fill=(0, 0, 0), font=font)
        
        # Add legend entries in two rows
        y_offset = 35
        legend_entries = [
            ("Cyan Arrows", (0, 255, 255), "Field Flow Direction"),
            ("Yellow Boxes", (255, 255, 0), "Magnified Regions of Interest"),
            ("Magenta Dots", (255, 0, 255), "Phase-Space Trajectory"),
            ("Heat Colors", (255, 128, 0), "Areas of Significant Change")
        ]
        
        # Position legend entries in two rows
        entry_width = legend_width // 2
        for i, (color_name, color_rgb, description) in enumerate(legend_entries):
            row = i // 2
            col = i % 2
            x_pos = col * entry_width + 10
            y_pos = y_offset + row * 30
            
            # Draw color sample
            draw.rectangle((x_pos, y_pos, x_pos + 20, y_pos + 20), fill=color_rgb, outline=(0, 0, 0))
            # Add text
            draw.text((x_pos + 25, y_pos), f"{color_name}: {description}", fill=(0, 0, 0), font=font)
        
        legend = np.array(legend_pil)
        legend_path = os.path.join(OUTPUT_DIR, "visualization_legend.pdf")
        numpy_to_pdf(legend, legend_path)
        logger.info(f"Visualization legend saved to {legend_path}")
        
        return final_visualizations
    
    except Exception as e:
        logger.error(f"Error enhancing field evolution: {e}")
        logger.error(traceback.format_exc())
        raise

    and phi-harmonic relationship markers.
    """
    # 1. Define golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 2. Create a dense 2-D meshgrid (x, t)
    x = np.linspace(0, 4 * np.pi, 1000)
    t = np.linspace(0, 2 * np.pi, 500)
    X, T = np.meshgrid(x, t)

    # Generate resonant field data
    resonant_field = make_resonant_field(phi, X, T)

    # 6. Define holographic colormap
    colors = ["#a8dadc", "#457b9d", "#1d3557", "#e63946"]
    n_bins = 256  # Increased for smoother gradients
    hues = np.linspace(0, 1, len(colors))**(1/phi)
    holographic_cmap = LinearSegmentedColormap.from_list("holographic", colors, N=n_bins)

    # 7. Enhanced plotting
    plt.figure(figsize=(12, 8), dpi=200, facecolor='black')

    # Add contourf three times at scales phi**0, phi**1, phi**2
    levels = np.linspace(resonant_field.min(), resonant_field.max(), 100)
    alphas = [0.3, 0.2, 0.15]
    for i, alpha in enumerate(alphas):
        plt.contourf(X, T, resonant_field / phi**i, levels=levels, cmap=holographic_cmap, alpha=alpha)

    # 8. Interference Overlay
    I = np.sin(X) * np.sin(X / phi)
    plt.imshow(I, extent=[0, 4*np.pi, 0, 2*np.pi], aspect='auto', origin='lower', cmap=holographic_cmap, alpha=0.15, interpolation='bilinear')

    # Mark phi-nodes: for i in range(-2, 4), plot vertical dashed lines at 2 * np.pi * phi**i
    for i in range(-2, 4):
        node_x = 2 * np.pi * phi**i
        plt.axvline(x=node_x, color='white', linestyle='--', alpha=0.2)

    # Add labels and title
    plt.title('Holographic Resonance Field Visualization', fontsize=16, color='white')
    plt.xlabel('Phase (radians)', fontsize=12, color='white')
    plt.ylabel('Time (radians)', fontsize=12, color='white')
    plt.colorbar(label='Amplitude', orientation='vertical', shrink=0.8)
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')

    # Apply golden spacing to x-ticks
    x_ticks = golden_spacing(5, np.pi)
    plt.xticks(x_ticks)

    # Add geometric component visualization
    # Add geometric component visualization
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    def plot_geometric_components(ax):
      """Creates an inset plot showing the spatial geometric patterns"""
      # Draw simplified representations of the three Platonic solids
      # Tetrahedron
      t_vertices = np.array([
          [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]
      ])
      t_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

      ax.add_collection3d(Poly3DCollection([t_vertices[f] for f in t_faces],
                          alpha=0.3, color='r', label='T₄'))

      # Add similar code for cube and dodecahedron

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.set_title('Geometric Basis')
      ax.legend()

    ax_inset = plt.axes([0.65, 0.15, 0.3, 0.3])
                plot_geometric_components(ax_inset)

                # Save the image
                output_path = os.path.join(OUTPUT_DIR, "frequency_analysis.png")
                plt.savefig(output_path, dpi=450, transparent=False, bbox_inches='tight', pad_inches=0.05)
                print("Image saved as 'frequency_analysis.png'")

                logger.info(f"Successfully enhanced holographic encoding visualization: {output_path}")
    # Find peak heights
    h1 = find_peak_height(fig_data, f1)
    h2 = find_peak_height(fig_data, f2)

    # Calculate midpoint for ratio label
    mid_f = (f1 + f2) / 2
    mid_h = min(h1, h2) * 0.8

    # Draw connecting arc
    theta = np.linspace(0, np.pi, 20)
    arc_x = mid_f + abs(f2-f1)/2 * np.cos(theta)
    arc_y = mid_h + abs(h2-h1)/4 * np.sin(theta)
    plt.plot(arc_x, arc_y, 'b-', alpha=0.5)

    # Add ratio text
    plt.text(mid_f, mid_h + abs(h2-h1)/4, ratio_text,
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

def find_peak_height(fig_data, freq):
    """Find the height of a peak at the given frequency"""
    # In a real implementation, find closest frequency and return its magnitude
    # For now, return a placeholder value
    idx = np.argmin(abs(np.array(fig_data['frequency_x']) - freq))
    return fig_data['frequency_y'][idx]

def plot_geometric_components(ax):

def add_ratio_marker(f1, f2, ratio_text, fig_data):
    if f_dodecahedron is None:
      f_dodecahedron = 741

    # Create enhanced visualization
    plt.figure(figsize=(12, 8))

    # Plot original spectrum data
    plt.plot(fig_data['frequency_x'], fig_data['frequency_y'], 'k-', alpha=0.7)

    # Add peak markers and labels
    plt.annotate('T₄', xy=(f_tetrahedron, find_peak_height(fig_data, f_tetrahedron)),
                 xytext=(0, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), fontsize=14)

    plt.annotate('C₈', xy=(f_cube, find_peak_height(fig_data, f_cube)),
                 xytext=(0, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), fontsize=14)

    plt.annotate('D₁₂', xy=(f_dodecahedron, find_peak_height(fig_data, f_dodecahedron)),
                 xytext=(0, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), fontsize=14)

    # Add ratio markers
    add_ratio_marker(f_tetrahedron, f_cube, "≈ 3:4", fig_data)
    add_ratio_marker(f_cube, f_dodecahedron, "≈ φ:1", fig_data)
    add_ratio_marker(f_tetrahedron, f_dodecahedron, "≈ 3φ:5", fig_data)

    # Add geometric component visualization
    ax_inset = plt.axes([0.65, 0.15, 0.3, 0.3])
    plot_geometric_components(ax_inset)

    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title('Enhanced Frequency Analysis with Platonic Solid Signatures', fontsize=16)

    # Add legend explaining the significance of the ratios
    plt.figtext(0.15, 0.15,
                "Phi-harmonic relationships:\n"
                "• 3:4 - Perfect fourth musical interval\n"
                "• φ:1 - Golden ratio (≈1.618:1)\n"
                "• 3φ:5 - Complex resonance pattern",
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def add_ratio_marker(f1, f2, ratio_text, fig_data):
    """Adds a visual marker showing the ratio between two frequencies"""
    # Find peak heights
    h1 = find_peak_height(fig_data, f1)
    h2 = find_peak_height(fig_data, f2)
    
    # Calculate midpoint for ratio label
    mid_f = (f1 + f2) / 2
    mid_h = min(h1, h2) * 0.8

    # Draw connecting arc
    theta = np.linspace(0, np.pi, 20)
    arc_x = mid_f + abs(f2-f1)/2 * np.cos(theta)
    arc_y = mid_h + abs(h2-h1)/4 * np.sin(theta)
    plt.plot(arc_x, arc_y, 'b-', alpha=0.5)
    
    # Add ratio text
    plt.text(mid_f, mid_h + abs(h2-h1)/4, ratio_text, 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

def find_peak_height(fig_data, freq):
    """Find the height of a peak at the given frequency"""
    # In a real implementation, find closest frequency and return its magnitude
    # For now, return a placeholder value
    idx = np.argmin(abs(np.array(fig_data['frequency_x']) - freq))
    return fig_data['frequency_y'][idx]

def plot_geometric_components(ax):
    """Creates an inset plot showing the spatial geometric patterns"""
    # Draw simplified representations of the three Platonic solids
    # Tetrahedron
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    t_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]
    ])
    t_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    ax.add_collection3d(Poly3DCollection([t_vertices[f] for f in t_faces],
                        alpha=0.3, color='r', label='T₄'))

    # Add similar code for cube and dodecahedron

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Geometric Basis')
    ax.legend()


def apply_mlx_gpu_processing(img_array):
    """
    Apply GPU-accelerated processing using MLX

    Args:
        img_array (numpy.ndarray): Input image array

    Returns:
        numpy.ndarray: Processed image array
    """
    try:
        start_time = time.time()

        if HAS_MLX:
            logger.info("Using MLX for GPU-accelerated processing")
            # Convert to MLX array for GPU processing
            mx_array = mx.array(img_array.astype(np.float32))

            # Normalize to [0, 1]
            mx_array = (mx_array - mx.min(mx_array)) / (mx.max(mx_array) - mx.min(mx_array))

            # Apply contrast enhancement
            mx_array = mx.power(mx_array, 1.5)

            # Apply additional sharpening (using built-in MLX operations)
            kernel = mx.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=mx.float32)
            kernel = kernel.reshape(3, 3, 1, 1)

            # Reshape for conv operation
            mx_reshaped = mx.transpose(mx_array, (2, 0, 1))  # [C, H, W]
            mx_reshaped = mx_reshaped.reshape(1, 3, mx_reshaped.shape[1], mx_reshaped.shape[2])

            # Apply convolution for each channel
            sharpened_channels = []
            for c in range(3):
                channel = mx_reshaped[:, c:c+1, :, :]
                sharpened = mx.conv2d(channel, kernel, padding='same')
                sharpened_channels.append(sharpened)

            # Combine channels and reshape back
            mx_sharp = mx.concatenate(sharpened_channels, axis=1)
            mx_sharp = mx_sharp[0]  # Remove batch dimension
            mx_sharp = mx.transpose(mx_sharp, (1, 2, 0))  # [H, W, C]

            # Clip values to valid range
            mx_sharp = mx.clip(mx_sharp, 0, 1)

            # Convert back to numpy and scale to [0, 255]
            result = (mx_sharp * 255).astype(np.uint8).numpy()
        else:
            logger.info("Using CPU processing (MLX not available)")
            # Fallback to CPU processing
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

            # Convert back to numpy and scale to [0, 255]
            result = (sharpened * 255).astype(np.uint8)

        processing_time = time.time() - start_time
        logger.info(f"Image processing completed in {processing_time:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error in GPU processing: {e}")
        logger.error(traceback.format_exc())
        # Fallback to simple enhancement
        enhanced = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
        return enhanced
    """
    Apply MLX-accelerated phase coherence analysis.
    
    Args:
        img (numpy.ndarray): Input image array
    
    Returns:
        numpy.ndarray: Phase coherence visualization with GPU acceleration
    """
    logger.info("Using MLX for phase coherence analysis")
    
    try:
        # Convert to MLX array
        mx_img = mx.array(img.astype(np.float32))
        
        # Convert to grayscale for phase analysis
        mx_gray = mx.mean(mx_img, axis=2, keepdims=True)
        
        # Apply FFT
        # Since MLX doesn't have direct FFT support, we'll reshape and process
        # Convert back to numpy, apply FFT, then back to MLX
        gray_np = mx_gray.numpy().squeeze()
        
        # Apply FFT using numpy
        f_transform = np.fft.fft2(gray_np)
        magnitude = np.abs(f_transform)
        phase = np.angle(f_transform)
        
        # Normalize phase to [0, 1] range
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        
        # Convert back to MLX for further processing
        mx_phase = mx.array(phase_normalized.astype(np.float32))
        
        # Apply colormap (plasma-like)
        # Create RGB mapping for phase values
        h, w = mx_phase.shape
        colored_phase = mx.zeros((h, w, 3), dtype=mx.float32)
        
        # Red channel (increases with phase)
        colored_phase = mx.concatenate([
            mx.power(mx_phase, 1.5),  # Red - higher in higher phase values
            mx.sin(mx_phase * 2 * np.pi),  # Green - cyclical with phase
            mx.cos(mx_phase * 2 * np.pi),  # Blue - cyclical, offset from green
        ], axis=2).reshape(h, w, 3)
        
        # Normalize to [0, 1]
        colored_phase = (colored_phase + 1) / 2
        
        # Apply edge detection to highlight boundaries
        # For edge detection, we'll use numpy/cv2 since MLX doesn't have direct support
        phase_np = (mx_phase.numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(phase_np, 50, 150)
        mx_edges = mx.array(edges.astype(np.float32)) / 255.0
        
        # Expand edges to 3 channels
        mx_edges = mx.expand_dims(mx_edges, axis=2)
        mx_edges = mx.concatenate([mx_edges, mx_edges, mx_edges], axis=2)
        
        # Combine phase colors with edge highlights
        alpha = 0.7
        result = mx.clip(colored_phase * alpha + mx_edges * (1-alpha), 0, 1)
        
        # Convert to uint8 and return
        return (result * 255).astype(np.uint8).numpy()
        
    except Exception as e:
        logger.error(f"Error in MLX phase coherence processing: {e}")
        logger.error(traceback.format_exc())
        # Fall back to CPU processing
        return apply_cpu_phase_coherence(img)

def apply_cpu_phase_coherence(img):
    """
    Apply CPU-based phase coherence analysis.
    
    Args:
        img (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Phase coherence visualization
    """
    logger.info("Using CPU for phase coherence analysis")
    
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
        logger.error(f"Error in CPU phase coherence processing: {e}")
        logger.error(traceback.format_exc())
        # Just return the original image if processing fails
        return img

def apply_mlx_interference(img):
    """
    Apply MLX-accelerated interference pattern analysis.
    
    Args:
        img (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Interference pattern visualization with GPU acceleration
    """
    logger.info("Using MLX for interference pattern analysis")
    
    try:
        # Convert to MLX array
        mx_img = mx.array(img.astype(np.float32))
        
        # Convert to grayscale for gradient analysis
        mx_gray = mx.mean(mx_img, axis=2)
        
        # For gradient analysis, we'll use numpy/cv2 then back to MLX
        gray_np = mx_gray.numpy()
        
        # Calculate gradients
        gradient_x = cv2.Sobel(gray_np, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_np, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # Normalize magnitude
        magnitude_normalized = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
        
        # Convert back to MLX
        mx_magnitude = mx.array(magnitude_normalized.astype(np.float32))
        mx_direction = mx.array(direction.astype(np.float32))
        
        # Create color mapping based on magnitude and direction
        h, w = mx_magnitude.shape
        interference = mx.zeros((h, w, 3), dtype=mx.float32)
        
        # Create plasma-like colormap based on magnitude
        # Red channel increases with magnitude
        interference = mx.stack([
            mx_magnitude,  # Red
            mx.sin(mx_direction) * mx_magnitude,  # Green
            mx.cos(mx_direction) * mx_magnitude,  # Blue
        ], axis=2)
        
        # Normalize to [0, 1]
        interference = (interference + 1) / 2
        
        # Convert to uint8 and return
        result = (interference * 255).astype(np.uint8).numpy()
        
        # Add directional arrows to show energy flow (using OpenCV)
        step = w // 20  # Number of arrows to display
        threshold = 0.5  # Only show arrows where magnitude is significant
        
        for y in range(step, h, step):
            for x in range(step, w, step):
                if magnitude_normalized[y, x] > threshold:
                    # Get direction angle
                    angle = direction[y, x]
                    # Arrow length based on magnitude
                    length = int(20 * magnitude_normalized[y, x])
                    # Calculate end point
                    end_x = int(x + length * np.cos(angle))
                    end_y = int(y + length * np.sin(angle))
                    # Ensure within bounds
                    end_x = min(max(end_x, 0), w-1)
                    end_y = min(max(end_y, 0), h-1)
                    # Draw arrow
                    cv2.arrowedLine(result, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.3)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in MLX interference pattern processing: {e}")
        logger.error(traceback.format_exc())
        # Fall back to CPU processing
        return apply_cpu_interference(img)

def apply_cpu_interference(img):
    """
    Apply CPU-based interference pattern analysis.
    
    Args:
        img (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Interference pattern visualization
    """
    logger.info("Using CPU for interference pattern analysis")
    
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
        
        # Apply colormap to magnitude
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
        
        # Add a legend for energy flow
        cv2.putText(colormap, "Energy Flow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return colormap
        
    except Exception as e:
        logger.error(f"Error in CPU interference pattern processing: {e}")
        logger.error(traceback.format_exc())
        # Just return the original image if processing fails
        return img

def apply_mlx_multiscale(img):
    """
    Apply MLX-accelerated multi-scale analysis.
    
    Args:
        img (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Multi-scale visualization with GPU acceleration
    """
    logger.info("Using MLX for multi-scale analysis")
    
    try:
        # For multi-scale analysis with geometric guides, we'll use a mix of MLX and numpy/OpenCV
        # First, process the image with MLX
        mx_img = mx.array(img.astype(np.float32))
        
        # Normalize
        mx_img = (mx_img - mx.min(mx_img)) / (mx.max(mx_img) - mx.min(mx_img))
        
        # Apply enhancement
        mx_img = mx.power(mx_img, 1.5)
        
        # Convert back to numpy for geometric overlays
        enhanced = (mx_img * 255).astype(np.uint8).numpy()
        
        # Create PIL image for drawing
        pil_img = Image.fromarray(enhanced)
        draw = ImageDraw.Draw(pil_img)
        
        # Get dimensions
        h, w = enhanced.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw multi-scale geometric guides
        scales = [1.0, 0.75, 0.5, 0.25]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        # Draw nested geometric patterns at different scales
        for scale, color in zip(scales, colors):
            size = int(min(w, h) * scale)
            
            # Draw square
            x1, y1 = center_x - size//2, center_y - size//2
            x2, y2 = center_x + size//2, center_y + size//2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw diagonal guides
            draw.line([x1, y1, x2, y2], fill=color, width=1)
            draw.line([x2, y1, x1, y2], fill=color, width=1)
            
            # Draw circles at corners to mark scale points
            radius = 5
            for corner in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
                draw.ellipse([corner[0]-radius, corner[1]-radius, 
                            corner[0]+radius, corner[1]+radius], 
                           fill=color)
        
        # Add magnified regions of interest
        regions = [
            ((w//4, h//4), "Self-Similarity Pattern"),
            ((3*w//4, h//4), "Scale Transition"),
            ((w//4, 3*h//4), "Phase Structure"),
            ((3*w//4, 3*h//4), "Information Distribution")
        ]
        
        for (x, y), label in regions:
            # Extract and magnify region
            size = 100
            x1, y1 = max(0, x-size//2), max(0, y-size//2)
            x2, y2 = min(w, x+size//2), min(h, y+size//2)
            
            if x2 > x1 and y2 > y1:  # Check if region is valid
                region = enhanced[y1:y2, x1:x2]
                if region.size > 0:
                    magnified = cv2.resize(region, (size*2, size*2))
                    
                    # Add border
                    cv2.rectangle(magnified, (0, 0), (size*2-1, size*2-1), 
                                (255, 255, 255), 2)
                    
                    # Add label
                    cv2.putText(magnified, label, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              (255, 255, 255), 1)
                    
                    # Insert magnified region
                    y_insert = y - size if y > h//2 else y + size//2
                    x_insert = x - size if x > w//2 else x + size//2
                    
                    y_insert = max(0, min(h - size*2, y_insert))
                    x_insert = max(0, min(w - size*2, x_insert))
                    
                    enhanced[y_insert:y_insert + size*2, 
                            x_insert:x_insert + size*2] = magnified
                        
                    # Draw connection line
                    cv2.line(enhanced, (x, y), 
                            (x_insert + size, y_insert + size),
                            (255, 255, 0), 1)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in MLX multi-scale analysis: {e}")
        logger.error(traceback.format_exc())
        # Fall back to CPU processing
        return apply_cpu_multiscale(img)

def apply_cpu_multiscale(img):
    """
    Apply CPU-based multi-scale analysis with geometric guides.
    
    Args:
        img (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Multi-scale visualization
    """
    logger.info("Using CPU for multi-scale analysis")
    
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
        
        return result
        
    except Exception as e:
        logger.error(f"Error in CPU multi-scale analysis: {e}")
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
        
        # Process original image with MLX/CPU
        if HAS_MLX:
            enhanced = apply_mlx_gpu_processing(img_array)
        else:
            enhanced = apply_cpu_processing_holographic(img_array)
        
        # Panel 1: Original with basic enhancement (top-left)
        final[:h, :w] = enhanced
        
        # Panel 2: Phase coherence visualization (top-right)
        if HAS_MLX:
            phase_panel = apply_mlx_phase_coherence(enhanced)
        else:
            phase_panel = apply_cpu_phase_coherence(enhanced)
        final[:h, w:] = phase_panel
        
        # Panel 3: Interference pattern analysis (bottom-left)
        if HAS_MLX:
            interference_panel = apply_mlx_interference(enhanced)
        else:
            interference_panel = apply_cpu_interference(enhanced)
        final[h:, :w] = interference_panel
        
        # Panel 4: Multi-scale analysis (bottom-right)
        if HAS_MLX:
            multiscale_panel = apply_mlx_multiscale(enhanced)
        else:
            multiscale_panel = apply_cpu_multiscale(enhanced)
        final[h:, w:] = multiscale_panel
        
        # Add labels to each panel
        final_pil = Image.fromarray(final)
        draw = ImageDraw.Draw(final_pil)
        
        try:
            font = ImageFont.truetype("Arial", 30)
        except IOError:
            font = ImageFont.load_default()
            
        # Add labels
        labels = [
            ("Original Pattern", (10, 10)),
            ("Phase Coherence Transitions", (w+10, 10)),
            ("Interference Pattern Dynamics", (10, h+10)),
            ("Multi-Scale Self-Similarity", (w+10, h+10))
        ]
        
        for text, pos in labels:
            # Add background for better text visibility
            bbox = draw.textbbox(pos, text, font=font)
            draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill="black")
            draw.text(pos, text, fill="white", font=font)
            
        final = np.array(final_pil)
        
        # Save the enhanced visualization
        output_path = os.path.join(OUTPUT_DIR, "holographic_encoding_enhanced.pdf")
        numpy_to_pdf(final, output_path)
        
        logger.info(f"Enhanced holographic encoding saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error enhancing holographic encoding: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main execution function for visualization enhancement."""
    logger.info("Starting visualization enhancement process...")
    
    try:
        start_time = time.time()
        
        # Check for input files
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
        if not os.path.exists(figures_dir):
            raise FileNotFoundError(f"Figures directory not found: {figures_dir}")
            
        # Process geometric basis visualization (Figure 2)
        geometric_basis_path = os.path.join(figures_dir, "geometric_basis.pdf")
        if os.path.exists(geometric_basis_path):
            enhanced_path = enhance_geometric_basis(geometric_basis_path)
            logger.info(f"Successfully enhanced geometric basis visualization: {enhanced_path}")
            
        # Process field evolution visualization (Figure 8)
        field_evolution_paths = [
            os.path.join(figures_dir, "field_evolution_0.pdf"),
            os.path.join(figures_dir, "field_evolution_8.pdf"),
            os.path.join(figures_dir, "field_evolution_17.pdf")
        ]
        if all(os.path.exists(path) for path in field_evolution_paths):
            enhanced_paths = enhance_field_evolution(field_evolution_paths)
            logger.info(f"Successfully enhanced field evolution visualizations: {enhanced_paths}")
            
        # Process holographic encoding visualization (Figure 9)
        # Process frequency analysis visualization (Figure 3)
        frequency_path = os.path.join(figures_dir, "frequency_analysis.pdf")
        if os.path.exists(frequency_path):
            print(f"Enhancing {frequency_path}...")

          # Make sure it can load data at all
            enhanced_path = enhance_frequency_analysis(frequency_path)
            logger.info(f"Successfully enhanced frequency analysis visualization: {enhanced_path}")
        # Process holographic encoding visualization (Figure 9)
        holographic_path = os.path.join(figures_dir, "holographic_encoding.pdf")
        if os.path.exists(holographic_path):
            enhanced_path = enhance_holographic_encoding(holographic_path)
            logger.info(f"Successfully enhanced holographic encoding visualization: {enhanced_path}")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def generate_harmonic(base_frequency, harmonic_number, phi):
    """Generates a harmonic frequency based on the golden ratio."""
    return base_frequency * (phi ** harmonic_number)

if __name__ == '__main__':
    # Set the harmonics here:

    f_tetrahedron = 396
    f_cube        = 528
    f_dodecahedron = 741

    # Test that the code runs appropriately here
    LOG_TESTING = False

    # Create a basic test
    if LOG_TESTING:
        logger.info(f"Testing that the freqs are outputted, T4: {f_tetrahedron} HZ, C8: {f_cube} Hz, D12: {f_dodecahedron}")

f_tetrahedron = None  # Placeholder for T4 frequency

        frequency_path = os.path.join(figures_dir, "frequency_analysis.pdf")
        if os.path.exists(frequency_path):
            print(f"Enhancing {frequency_path}...")
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            print("generating harmonic functions, labels, frequencies, from t4,c8 and d12")
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            from PIL import Image, ImageDraw, ImageFont
            def golden_spacing(n, base_step):
              #Returning -scaled sequence for n steps
              phi = (1 + np.sqrt(5)) / 2
              return [base_step * phi**k for k in range(n)]


            def plot_geometric_components(ax):
                """Creates an inset plot showing the spatial geometric patterns"""
                # Draw simplified representations of the three Platonic solids
                # Tetrahedron

                t_vertices = np.array([
                    [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]
                ])
                t_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

                ax.add_collection3d(Poly3DCollection([t_vertices[f] for f in t_faces],
                                    alpha=0.3, color='r', label='T₄'))

                # Add similar code for cube and dodecahedron

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Geometric Basis')
                ax.legend(loc = 'upper right')

            print("The image generated may be different")

            h_path = os.path.join(figures_dir, 'holographic_encoding.pdf')
            print("This is to ensure it is outputed from file ",h_path)

            def enhance_holographic_encoding(output_path):
                  img_path = 'holographic_encoding.png'
                  #f_tetrahedron = 396
                  #f_cube = 528
                  #f_dodecahedron = 741
                  fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': '3d'})
                  ax.set_xlabel("X")
                  ax.set_ylabel("Y")
                  ax.set_zlabel("Z")
                  ax.set_title("Harmonic Frequencies of Geometric Formations")
                  x = np.linspace(0, 50, 500)
                  y = np.sin(x)
                  z = np.cos(x)
                  ax.plot(x, y, z, label='Harmonic spiral')
                  ax.legend()
                  import os
                  h_path = os.path.join(OUTPUT_DIR, img_path)
                  #ax.plot(y, z)
                  plt.savefig(h_path)
                  print ("made figure 9 to ", h_path)

                  #This section adds text about the harmonics and their ratios"
                  return ""
        



          # Add similar code for cube and dodecahedron

          ax.set_xlabel('X')
          ax.set_ylabel('Y')
          ax.set_zlabel('Z')
          ax.set_title('Geometric Basis')
          ax.legend()


    # Save the image
    output_path = os.path.join(OUTPUT_DIR, "holographic_field.png")
                  ax.set_zlabel('Z')
                  ax.set_title('Geometric Basis')
                  ax.legend()


                # Save the image
                output_path = os.path.join(OUTPUT_DIR, "holographic_field.png")
            f.write("=====================================================\n\n")
            f.write("To update your LaTeX document with the enhanced figures, make the following changes:\n\n")
            
            # Figure 2 updates
            f.write("1. For Figure 2 (Geometric Basis):\n")
            f.write("   Original: \\includegraphics[width=\\textwidth]{figures/geometric_basis.pdf}\n")
            f.write(f"   Updated: \\includegraphics[width=\\textwidth]{os.path.relpath(os.path.join(OUTPUT_DIR, 'geometric_basis_enhanced.pdf'), os.path.dirname(os.path.abspath(__file__)))}\n\n")
            
            # Figure 8 updates
            f.write("2. For Figure 8 (Field Evolution):\n")
            f.write("   You can either replace the individual images:\n")
            f.write("   - Replace: field_evolution_0.pdf with field_evolution_0_enhanced.pdf\n")
            f.write("   - Replace: field_evolution_8.pdf with field_evolution_8_enhanced.pdf\n")
            f.write("   - Replace: field_evolution_17.pdf with field_evolution_17_enhanced.pdf\n\n")
            f.write("   OR use the combined visualization for a clearer comparison:\n")
            f.write("   \\begin{figure}[H]\n")
            f.write("       \\centering\n")
            f.write(f"       \\includegraphics[width=\\textwidth]{os.path.relpath(os.path.join(OUTPUT_DIR, 'field_evolution_combined_enhanced.pdf'), os.path.dirname(os.path.abspath(__file__)))}\n")
            f.write("       \\caption{Enhanced visualization of field evolution showing clear stage differences with color-coded highlighting of changes.}\n")
            f.write("       \\label{fig:field_evolution_enhanced}\n")
            f.write("   \\end{figure}\n\n")
            
            # Figure 9 updates
            f.write("3. For Figure 9 (Holographic Encoding):\n")
            f.write("   Replace:\n")
            f.write("   \\includegraphics[width=\\textwidth]{figures/holographic_encoding.pdf}\n")
            f.write("   With:\n")
            f.write(f"   \\includegraphics[width=\\textwidth]{os.path.relpath(os.path.join(OUTPUT_DIR, 'holographic_encoding_enhanced.pdf'), os.path.dirname(os.path.abspath(__file__)))}\n\n")
            
            # Add visualization legend
            f.write("4. Consider adding the legend for clarity:\n")
            f.write("   \\begin{figure}[H]\n")
            f.write("       \\centering\n")
            f.write(f"       \\includegraphics[width=0.7\\textwidth]{os.path.relpath(os.path.join(OUTPUT_DIR, 'visualization_legend.pdf'), os.path.dirname(os.path.abspath(__file__)))}\n")
            f.write("       \\caption{Legend for enhanced visualizations showing the meaning of colors and visual elements.}\n")
            f.write("       \\label{fig:viz_legend}\n")
            f.write("   \\end{figure}\n")
            
        # Print completion information
        total_time = time.time() - start_time
        logger.info(f"Visualization enhancement completed in {total_time:.2f} seconds")
        logger.info(f"Enhanced figures saved to: {OUTPUT_DIR}")
        logger.info(f"LaTeX update guide created at: {latex_guide}")
        
        # Calculate the number of generated files
        enhanced_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('_enhanced.pdf')])
        logger.info(f"Generated {enhanced_files:8d} enhanced visualization files.")
        
        # Try to open the output directory for the user
        logger.info("Opening the output directory...")
        if sys.platform == "darwin":  # macOS
            os.system(f"open {OUTPUT_DIR}")
        elif sys.platform == "win32":  # Windows
            os.system(f"explorer {OUTPUT_DIR}")
        elif sys.platform == "linux":  # Linux
            os.system(f"xdg-open {OUTPUT_DIR}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error in visualization enhancement process: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())

