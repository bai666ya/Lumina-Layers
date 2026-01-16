"""
Lumina Calibration Tool
-----------------------
Part of the Lumina-Layers Ecosystem.

This tool extracts color data from a physical calibration print to generate
a 'Ground Truth' Look-Up Table (LUT) for the printer.

Key Features:
- Perspective Warp & Lens Distortion Correction
- Auto White Balance & Vignette Correction
- Physical Grid Mapping (34x34 source -> 32x32 data)
- Simulated Optical Mixing Preview (Beer-Lambert Approximation)
- Clean UI without overlay obstructions

Author: [MIN]
License: CC BY-NC-SA 4.0
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os

# --- Configuration & Constants ---

# The physical print has a border, making it 34x34 cells.
# We only care about the inner 32x32 data area.
PHYSICAL_GRID_SIZE = 34
DATA_GRID_SIZE = 32

# Internal resolution for image processing
DST_SIZE = 1000
CELL_SIZE = DST_SIZE / PHYSICAL_GRID_SIZE

# Path to save the extracted LUT
LUT_FILE_PATH = os.path.join(tempfile.gettempdir(), "my_printer_lut.npy")


# --- 1. Simulation Engine ---

def generate_simulated_reference():
    """
    Generates a theoretical reference image of what the calibration board
    SHOULD look like, based on simple optical mixing logic.
    Used for visual comparison by the user.
    """
    # Base colors (Approximate values for White, Red, Yellow, Blue filaments)
    colors = {
        0: np.array([250, 250, 250]),  # White
        1: np.array([220, 20, 60]),    # Red
        2: np.array([255, 230, 0]),    # Yellow
        3: np.array([0, 100, 240])     # Blue
    }
    
    ref_img = np.zeros((DATA_GRID_SIZE, DATA_GRID_SIZE, 3), dtype=np.uint8)

    # Generate 1024 unique stack combinations
    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        
        # Stack order: Top -> Bottom (Face Down printing logic)
        stack = digits[::-1]

        # Simple averaging to simulate transmission
        # (Real physics is handled in the main Engine, this is just for UI alignment)
        mixed = np.zeros(3, dtype=float)
        for mid in stack: 
            mixed += colors[mid]
        mixed /= 5.0

        row, col = i // DATA_GRID_SIZE, i % DATA_GRID_SIZE
        ref_img[row, col] = mixed.astype(np.uint8)

    # Upscale for better visibility
    return cv2.resize(ref_img, (512, 512), interpolation=cv2.INTER_NEAREST)


# --- 2. Image Processing Core ---

def rotate_image_logic(img, angle_mode):
    """Rotates the input image by 90 degrees."""
    if img is None: return None
    if angle_mode == "Left (-90°)":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle_mode == "Right (+90°)":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def draw_points(img, points):
    """Draws annotation markers on the image for the 4 corner calibration."""
    if img is None: return None
    vis = img.copy()
    
    # Standard sequence: White -> Red -> Blue -> Yellow
    labels = ["TL (White)", "TR (Red)", "BR (Blue)", "BL (Yellow)"]
    colors = [(255, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 255)]

    for i, pt in enumerate(points):
        # Green for points beyond the required 4
        color = colors[i] if i < 4 else (0, 255, 0)
        
        # Draw dot
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 15, color, -1)
        # Draw outline
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 15, (0, 0, 0), 2)
        # Draw Number
        cv2.putText(vis, str(i + 1), (int(pt[0]) + 20, int(pt[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Draw Label for the first 4 points
        if i < 4:
            cv2.putText(vis, labels[i], (int(pt[0]) + 20, int(pt[1]) + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return vis


def apply_auto_white_balance(img):
    """
    Corrects color temperature by assuming the 4 corners of the image
    (where the white blocks should be) are neutral white.
    """
    h, w, _ = img.shape
    m = 50 # Margin size for sampling
    
    # Sample 4 corners
    tl = img[0:m, 0:m].mean(axis=(0, 1))
    tr = img[0:m, w - m:w].mean(axis=(0, 1))
    bl = img[h - m:h, 0:m].mean(axis=(0, 1))
    br = img[h - m:h, w - m:w].mean(axis=(0, 1))
    
    # Calculate gain
    avg_white = (tl + tr + bl + br) / 4.0
    gain = np.array([255, 255, 255]) / (avg_white + 1e-5)
    
    # Apply
    return np.clip(img.astype(float) * gain, 0, 255).astype(np.uint8)


def apply_brightness_correction(img):
    """
    Corrects uneven lighting (vignette) by creating a gradient mask
    based on the brightness of the 4 corners.
    """
    h, w, _ = img.shape
    # Convert to LAB to operate on Luminance (L) only
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    
    m = 50
    tl = l[0:m, 0:m].mean()
    tr = l[0:m, w - m:w].mean()
    bl = l[h - m:h, 0:m].mean()
    br = l[h - m:h, w - m:w].mean()
    
    # Create gradient planes
    top = np.linspace(tl, tr, w)
    bot = np.linspace(bl, br, w)
    
    # Interpolate between top and bottom
    mask = np.array([top * (1 - y / h) + bot * (y / h) for y in range(h)])
    
    target = (tl + tr + bl + br) / 4.0
    l_new = np.clip(l.astype(float) * (target / (mask + 1e-5)), 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(cv2.merge([l_new, a, b]), cv2.COLOR_LAB2RGB)


def run_calibration(img, points, offset_x, offset_y, zoom, barrel, wb, bright):
    """
    Main extraction pipeline:
    1. Perspective Warp (Mapping 4 physical corners to the 34x34 grid)
    2. Image Enhancements (WB/Brightness)
    3. Grid Sampling (Extracting 32x32 center data)
    """
    if img is None: return None, None, None, "❌ Please upload an image first."
    if len(points) != 4: return None, None, None, "❌ Please click exactly 4 corner points."

    # 1. Perspective Transform
    # We map the 4 user points to the centers of the corner cells in the 34x34 grid.
    half = CELL_SIZE / 2.0
    src = np.float32(points)
    dst = np.float32([
        [half, half],                         # Top-Left
        [DST_SIZE - half, half],              # Top-Right
        [DST_SIZE - half, DST_SIZE - half],   # Bottom-Right
        [half, DST_SIZE - half]               # Bottom-Left
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (DST_SIZE, DST_SIZE))

    # 2. Enhancements
    if wb: warped = apply_auto_white_balance(warped)
    if bright: warped = apply_brightness_correction(warped)

    # 3. Sampling
    extracted = np.zeros((DATA_GRID_SIZE, DATA_GRID_SIZE, 3), dtype=np.uint8)
    vis = warped.copy()

    for r in range(DATA_GRID_SIZE):
        for c in range(DATA_GRID_SIZE):
            # Map logical index (0-31) to physical grid index (1-32)
            # skipping the border ring
            phys_r, phys_c = r + 1, c + 1
            
            # Normalize coordinates (-1 to 1)
            nx = (phys_c + 0.5) / PHYSICAL_GRID_SIZE * 2 - 1
            ny = (phys_r + 0.5) / PHYSICAL_GRID_SIZE * 2 - 1

            # Barrel Distortion Correction
            rad = np.sqrt(nx ** 2 + ny ** 2)
            k = 1 + barrel * (rad ** 2)
            dx, dy = nx * k * zoom, ny * k * zoom

            # Map back to pixel coordinates
            cx = (dx + 1) / 2 * DST_SIZE + offset_x
            cy = (dy + 1) / 2 * DST_SIZE + offset_y

            # Sample if within bounds
            if 0 <= cx < DST_SIZE and 0 <= cy < DST_SIZE:
                x0, y0 = int(max(0, cx - 4)), int(max(0, cy - 4))
                x1, y1 = int(min(DST_SIZE, cx + 4)), int(min(DST_SIZE, cy + 4))
                
                reg = warped[y0:y1, x0:x1]
                avg = reg.mean(axis=(0, 1)).astype(int) if reg.size > 0 else [0, 0, 0]
                
                # Visual Marker
                cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 0),
                               cv2.MARKER_CROSS, 8, 1)
            else:
                avg = [0, 0, 0]
            
            extracted[r, c] = avg

    # Save and Return
    np.save(LUT_FILE_PATH, extracted)
    prev = cv2.resize(extracted, (512, 512), interpolation=cv2.INTER_NEAREST)
    return vis, prev, LUT_FILE_PATH, "✅ Extraction Complete"


def probe_file(evt: gr.SelectData):
    """Allows user to click the LUT preview to inspect RGB values."""
    if not os.path.exists(LUT_FILE_PATH): return "⚠️ No Data", None, None
    try:
        lut = np.load(LUT_FILE_PATH)
    except:
        return "⚠️ Corrupted", None, None

    x, y = evt.index
    # Map click coordinates back to 32x32 grid
    scale = 512 / DATA_GRID_SIZE
    c = min(max(int(x / scale), 0), DATA_GRID_SIZE - 1)
    r = min(max(int(y / scale), 0), DATA_GRID_SIZE - 1)

    rgb = lut[r, c]
    hex_c = '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    html = f"""
    <div style='background: #f0f0f0; padding: 5px; border-radius: 5px;'>
        <b>R{r + 1} / C{c + 1}</b><br>
        <div style='background:{hex_c}; width:50px; height:20px; border:1px solid #000; display:inline-block; vertical-align:middle;'></div>
        <span style='margin-left:5px;'>{hex_c}</span>
    </div>
    """
    return html, hex_c, (r, c)


def manual_fix(coord, color_input):
    """Allows manual override of a specific cell color."""
    if not coord or not os.path.exists(LUT_FILE_PATH): return None, "⚠️ Error"
    
    try:
        lut = np.load(LUT_FILE_PATH)
        r, c = coord
        
        # [修复] 健壮的颜色解析逻辑
        new_color = [0, 0, 0]
        
        # 情况1: 如果是 'rgb(r, g, b)' 格式
        if str(color_input).startswith('rgb'):
            # 简单的字符串清洗
            clean_str = str(color_input).replace('rgb', '').replace('a', '').replace('(', '').replace(')', '')
            parts = clean_str.split(',')
            if len(parts) >= 3:
                new_color = [int(float(p.strip())) for p in parts[:3]]
        
        # 情况2: 如果是 '#RRGGBB' 格式
        elif str(color_input).startswith('#'):
            hex_s = color_input.lstrip('#')
            new_color = [int(hex_s[i:i + 2], 16) for i in (0, 2, 4)]
            
        else:
            # 尝试直接按 Hex 解析
            hex_s = color_input
            new_color = [int(hex_s[i:i + 2], 16) for i in (0, 2, 4)]

        lut[r, c] = new_color
        np.save(LUT_FILE_PATH, lut)
        return cv2.resize(lut, (512, 512), interpolation=cv2.INTER_NEAREST), "✅ Patched"
        
    except Exception as e:
        print(f"Color patch error: {e}")
        return None, f"❌ Format Error: {color_input}"


# --- 3. UI Layout ---

with gr.Blocks(title="Lumina Calibration") as app:
    gr.Markdown("## 🧪 Lumina Calibration Tool")
    gr.Markdown("Extracts color DNA from your printer. Align the grid and generate your unique LUT.")

    state_img = gr.State(None)
    state_pts = gr.State([])
    curr_coord = gr.State(None)
    ref_img = generate_simulated_reference()

    with gr.Row():
        # --- LEFT COLUMN: Input & Controls ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Photo (Face Up)")
            img_in = gr.Image(show_label=False, type="numpy", interactive=True)

            with gr.Row():
                rot_btn = gr.Button("↺ Rotate")
                clear_btn = gr.Button("🗑️ Reset Points")

            gr.Markdown("### 3. Geometry Fix")
            with gr.Row():
                wb = gr.Checkbox(label="Auto White Balance", value=True)
                bf = gr.Checkbox(label="Vignette Fix", value=False)

            zoom = gr.Slider(0.8, 1.2, 1.0, step=0.005, label="Zoom")
            barrel = gr.Slider(-0.2, 0.2, 0.0, step=0.01, label="Distortion")
            off_x = gr.Slider(-30, 30, 0, step=1, label="Offset X")
            off_y = gr.Slider(-30, 30, 0, step=1, label="Offset Y")

            run_btn = gr.Button("🚀 Run Extraction", variant="primary")
            log = gr.Textbox(label="Log", lines=1)

        # --- RIGHT COLUMN: Visual Feedback ---
        with gr.Column(scale=1):
            hint = gr.Markdown("### 👉 Click: **White Block (Top-Left)**")
            work_img = gr.Image(show_label=False, interactive=True)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 4. Your Result")
                    warp_view = gr.Image(show_label=False, interactive=True)
                with gr.Column():
                    gr.Markdown("### 🎯 Reference")
                    ref_view = gr.Image(show_label=False, value=ref_img, interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 5. Final LUT (Click to Fix)")
                    lut_view = gr.Image(show_label=False, interactive=True)
                
                with gr.Column():
                    gr.Markdown("### 🛠️ Patch Tool")
                    probe_html = gr.HTML("Click a cell on the left...")
                    picker = gr.ColorPicker(label="Override Color", value="#FF0000")
                    fix_btn = gr.Button("🔧 Apply Fix")
                    dl_btn = gr.File(label="Download .npy")

    # --- Event Handling ---
    
    # 1. Upload & Reset
    def on_up(i): return i, i, [], None, "### 👉 Click: **White Block (Top-Left)**"
    img_in.upload(on_up, img_in, [state_img, work_img, state_pts, curr_coord, hint])
    
    # 2. Rotate
    def on_rot(i): 
        if i is None: return None, None, []
        r = rotate_image_logic(i, "Left (-90°)")
        return r, r, []
    rot_btn.click(on_rot, state_img, [state_img, work_img, state_pts])
    
    # 3. Click Points
    def on_clk(img, pts, evt: gr.SelectData):
        if len(pts) >= 4: return img, pts, "### ✅ Positioning Complete"
        n = pts + [[evt.index[0], evt.index[1]]]
        vis = draw_points(img, n)
        lbls = ["Red Block (Top-Right)", "Blue Block (Bottom-Right)", "Yellow Block (Bottom-Left)", "✅ Ready"]
        return vis, n, f"### 👉 Click: **{lbls[len(n)-1]}**"
    work_img.select(on_clk, [state_img, state_pts], [work_img, state_pts, hint])
    
    # 4. Clear
    clear_btn.click(lambda i: (i, [], "### 👉 Click: **White Block**"), state_img, [work_img, state_pts, hint])

    # 5. Run Extraction
    # IMPORTANT: Variable 'bf' (checkbox) maps to function argument 'bright'
    # Use variable name 'bf' in the inputs list
    extract_inputs = [state_img, state_pts, off_x, off_y, zoom, barrel, wb, bf]
    extract_outputs = [warp_view, lut_view, dl_btn, log]
    
    run_btn.click(run_calibration, extract_inputs, extract_outputs)
    
    # Live Preview on slider change
    for s in [off_x, off_y, zoom, barrel]: 
        s.release(run_calibration, extract_inputs, extract_outputs)

    # 6. Manual Patching
    lut_view.select(probe_file, [], [probe_html, picker, curr_coord])
    fix_btn.click(manual_fix, [curr_coord, picker], [lut_view, log])

if __name__ == "__main__":
    app.launch(inbrowser=True)
