#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.colors as mcolors # Needed for comparison figure normalization

import cv2
import numpy as np
import base64
import io
import os
import datetime
import traceback # For detailed error printing
import json # For checking figure structure

# --- Image Processing Functions ---
def apply_frequency_filter(image, filter_mask):
    """Applies a given filter mask in the frequency domain."""
    if image is None or image.ndim != 2: print("Error: apply_frequency_filter requires a 2D grayscale image."); return None
    try:
        # Ensure input is float32 for FFT
        dft = np.fft.fft2(image.astype(np.float32)); dft_shifted = np.fft.fftshift(dft)
        # Apply mask
        filtered_dft_shifted = dft_shifted * filter_mask; filtered_dft = np.fft.ifftshift(filtered_dft_shifted)
        # Inverse FFT
        filtered_image = np.fft.ifft2(filtered_dft); filtered_image = np.abs(filtered_image)
        # Normalize to 0-255 and convert to uint8
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
        return filtered_image.astype(np.uint8)
    except Exception as e: print(f"Error during frequency filtering: {e}"); traceback.print_exc(); return None

def create_gaussian_lowpass_filter(shape, cutoff_freq):
    """Creates a Gaussian Low Pass Filter mask."""
    rows, cols = shape; center_row, center_col = rows // 2, cols // 2
    x, y = np.arange(cols), np.arange(rows); xx, yy = np.meshgrid(x, y)
    distance = np.sqrt((xx - center_col)**2 + (yy - center_row)**2)
    # Add epsilon to avoid division by zero if cutoff_freq is very close to 0
    mask = np.exp(-(distance**2) / (2 * (cutoff_freq**2 + 1e-6)))
    return mask

def create_gaussian_highpass_filter(shape, cutoff_freq):
    """Creates a Gaussian High Pass Filter mask."""
    # A GHPS is 1 - GLPF
    glpf_mask = create_gaussian_lowpass_filter(shape, cutoff_freq)
    ghpf_mask = 1 - glpf_mask
    return ghpf_mask

def segment_image_otsu(image):
    """ Performs Otsu's thresholding and finds contours. """
    if image is None or image.ndim != 2: print("Error: segment_image_otsu requires a 2D grayscale image."); return 0, None, []
    try:
        # Apply Otsu's thresholding
        T, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return T, mask, contours
    except Exception as e: print(f"Error during segmentation: {e}"); traceback.print_exc(); return 0, None, []

# --- Helper Functions for Dash ---

def parse_contents(contents):
    """Decodes image upload contents into an OpenCV image."""
    if contents is None: return None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Convert to numpy array from buffer
        buffer = np.frombuffer(decoded, dtype=np.uint8)
        # Decode the numpy array as a grayscale image
        img_cv = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            print("Warning: cv2.imdecode returned None. Invalid image format or data.")
            return None
        return img_cv
    except Exception as e: print(f"Error parsing contents: {e}"); traceback.print_exc(); return None

# --- Create Plotly Figure Helper ---
def create_plotly_fig(img_array, title="Image"):
    """Converts an OpenCV image (NumPy array) to a Plotly figure using Heatmap."""
    fig = None # Initialize fig to None
    # Check if the image array is valid (2D grayscale)
    if img_array is not None and img_array.ndim == 2 and img_array.size > 0:
        try:
            # Create a heatmap figure
            fig = go.Figure(go.Heatmap(z=img_array, colorscale='gray', showscale=False))
            # Configure layout for image display: reverse y-axis, fix aspect ratio
            fig.update_layout(yaxis={'autorange': 'reversed', 'scaleanchor': 'x', 'scaleratio': 1}, xaxis={'constrain': 'domain'})
        except Exception as e:
            print(f"!!! EXCEPTION creating Figure with go.Heatmap for '{title}': {e}"); traceback.print_exc(); fig = None
    # If figure creation failed or no valid image, create an empty figure
    if fig is None: fig = go.Figure()
    # Set a final title, indicating if image data is missing
    final_title = title
    if not (img_array is not None and img_array.size > 0) and "Upload" not in title and "Click" not in title and "Apply" not in title:
        final_title = f"{title} (No Image Data)"
    # Update general layout appearance (dark theme, centered title, margins)
    fig.update_layout(
        title=dict(text=final_title, x=0.5, xanchor='center'),
        plot_bgcolor='#1a1a1a', paper_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0'),
        margin=dict(l=5, r=5, t=40, b=5) # Minimal margins
    )
    # Hide axes ticks and gridlines
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    # Final check: Try converting to JSON to catch potential Plotly errors
    try:
        fig_json = fig.to_json()
    except Exception as e:
        print(f"!!! ERROR converting figure to JSON for '{title}': {e}"); traceback.print_exc()
        # Return a minimal error figure if JSON conversion fails
        fig = go.Figure().update_layout(title=f"{title} (Figure Error)")
    return fig

# --- Create Comparison Figure Helper ---
def create_comparison_figure(img_orig, img_enhanced, enh_filename_short, cmap_diff='plasma', contour_color_rgb=(255, 255, 0)):
    """Generates a 1x2 comparison Plotly figure showing only Difference and Segmentation."""
    # --- Input Validation ---
    if img_orig is None or img_enhanced is None:
        print("!!! ERROR in create_comparison_figure: Input image(s) are None.")
        return create_plotly_fig(None, "Comparison Data Unavailable")
    if img_orig.shape != img_enhanced.shape:
        print(f"!!! ERROR in create_comparison_figure: Shape mismatch! Original {img_orig.shape} vs Enhanced {img_enhanced.shape}")
        return create_plotly_fig(None, f"Error: Shape Mismatch {img_orig.shape} vs {img_enhanced.shape}")
    if img_orig.ndim != 2 or img_enhanced.ndim != 2:
        print(f"!!! ERROR in create_comparison_figure: Input image(s) not 2D grayscale.")
        return create_plotly_fig(None, "Error: Inputs must be grayscale")

    print(f"--- create_comparison_figure (1x2): Starting. Orig shape={img_orig.shape}, Enh shape={img_enhanced.shape} ---")
    fig = None
    try:
        # 1. Calculate Absolute Difference
        orig_float = img_orig.astype(np.float32); enh_float = img_enhanced.astype(np.float32)
        abs_difference = np.abs(enh_float - orig_float)
        print(f"--- create_comparison_figure: Difference calculated. Shape={abs_difference.shape}, dtype={abs_difference.dtype} ---")

        # 2. Perform Segmentation on Enhanced Image
        print(f"--- create_comparison_figure: Performing segmentation on enhanced image... ---")
        otsu_threshold, binary_mask, contours = segment_image_otsu(img_enhanced)
        num_contours = 0
        # Create a BGR version to draw colorful contours
        img_enhanced_bgr_for_contours = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)
        if binary_mask is not None: # Check if segmentation was successful
            num_contours = len(contours)
            contour_color_bgr = tuple(contour_color_rgb[::-1]); # Convert RGB to BGR for OpenCV
            print(f"--- create_comparison_figure: Drawing {num_contours} contours... ---")
            cv2.drawContours(img_enhanced_bgr_for_contours, contours, -1, contour_color_bgr, 2) # Draw contours
        else:
             print(f"!!! WARNING in create_comparison_figure: Segmentation failed (otsu returned None mask). Displaying without contours.")

        # Convert BGR image (with contours) to RGB for Plotly go.Image
        img_enhanced_rgb_for_plotly = cv2.cvtColor(img_enhanced_bgr_for_contours, cv2.COLOR_BGR2RGB)
        print(f"--- create_comparison_figure: Converted contoured image to RGB. Shape={img_enhanced_rgb_for_plotly.shape}, dtype={img_enhanced_rgb_for_plotly.dtype} ---")

        # 3. Create Plotly Subplots (1 Row, 2 Cols)
        fig = sp.make_subplots( rows=1, cols=2, subplot_titles=("Absolute Difference", "Segmentation"), vertical_spacing=0.08, horizontal_spacing=0.05 )
        print("--- create_comparison_figure (1x2): Subplots created. Adding traces... ---")

        # Add Traces
        print(f"--- create_comparison_figure: Adding Heatmap trace (Diff)... ---")
        # Use nested dict for colorbar title and side
        fig.add_trace(go.Heatmap(z=abs_difference, colorscale=cmap_diff, colorbar=dict(title=dict(text="Diff Mag", side="right"), thickness=15, len=0.8, y=0.5), showscale=True), row=1, col=1)

        print(f"--- create_comparison_figure: Adding Image trace (Segmentation)... ---")
        fig.add_trace(go.Image(z=img_enhanced_rgb_for_plotly), row=1, col=2) # Use the RGB image
        print("--- create_comparison_figure (1x2): All 2 traces added. ---")

        # Add Annotations
        # Use domain referencing for annotation positioning robustness
        fig.add_annotation( text=f"Otsu Thresh: {otsu_threshold:.1f}<br>Contours: {num_contours}", align='left', showarrow=False, x=0.02, y=0.02, bgcolor="rgba(0,0,0,0.65)", bordercolor="#cccccc", borderwidth=1, font=dict(color="#ffffff", size=10), xref="x2 domain", yref="y2 domain" )
        print("--- create_comparison_figure (1x2): Annotation added. ---")

        # Update Layout
        fig_bg_color = '#1a1a1a'; text_color = '#e0e0e0'; accent_text_color = '#ffffff'
        fig.update_layout(
            plot_bgcolor=fig_bg_color, paper_bgcolor=fig_bg_color,
            font=dict(color=text_color),
            title_text=f"Comparison Analysis: Difference & Segmentation",
            title_font=dict(size=18, color=accent_text_color), title_x=0.5,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # Apply axis settings to hide ticks/grid and fix aspect ratio
        axis_settings = {'visible': False, 'showgrid': False, 'zeroline': False}
        fig.update_xaxes(**axis_settings)
        fig.update_yaxes(**axis_settings)
        yaxis_img_settings = {'scaleratio': 1, 'autorange': 'reversed'}
        fig.update_yaxes(row=1, col=1, scaleanchor='x1', **yaxis_img_settings) # Anchor y1 to x1
        fig.update_yaxes(row=1, col=2, scaleanchor='x2', **yaxis_img_settings) # Anchor y2 to x2
        print("--- create_comparison_figure (1x2): Layout updated. ---")

        # Final check: Try converting to JSON
        fig_json_comp = fig.to_json();
        print(f"--- create_comparison_figure (1x2): Final check (to_json) OK. Traces: {len(fig.data)} ---")

    except Exception as e:
        print(f"!!! EXCEPTION in create_comparison_figure: {e}"); traceback.print_exc()
        fig = create_plotly_fig(None, "Error Generating Comparison") # Fallback error figure
    return fig

# --- Dash App Initialization ---
# Use Cyborg dark theme and allow suppressed exceptions for dynamic callbacks
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server # Expose server variable for potential deployment (e.g., Gunicorn)

# --- App Layout (RESPONSIVE) ---
app.layout = dbc.Container([
    # --- Text Header ---
    dbc.Row(
        dbc.Col(
            html.H1(
                "Group Project 🔬",
                className="text-center text-secondary mb-4" # Centered, secondary gray, margin bottom
            ),
            width=12 # Header always takes full width
        )
    ),
    # --- Main Content Row ---
    dbc.Row([
        # Controls Column
        # Takes 4/12 width on large screens (lg), full width (12/12) on medium and smaller
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Controls"),
                dbc.CardBody([
                    dcc.Upload(id='upload-image', children=html.Div(['Drag and Drop or ', html.A('Select Ultrasound Image')]), style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '15px'}, multiple=False),
                    dbc.Label("Enhancement Technique:", html_for="filter-type"),
                    dbc.RadioItems(options=[{'label': 'Smoothing (Low Pass)', 'value': 'smooth'}, {'label': 'Sharpening (High Pass)', 'value': 'sharpen'}, {'label': 'High-Boost Sharpening', 'value': 'highboost'}], value='highboost', id="filter-type", inline=False, className="mb-3"),
                    html.Div(id='parameter-input-div'), # Dynamic controls appear here
                    dbc.Button("Apply Enhancement", id="enhance-button", color="primary", className="mt-3 me-2"),
                    dbc.Button("Compare Images", id="compare-button", color="secondary", className="mt-3", disabled=True),
                    html.Div(id="status-message", className="mt-3 text-info small"), # Status messages
                ])
            ]),
            lg=4, md=12, # Responsive width: 4 on large, 12 (full) on medium/small
            className="mb-3 mb-lg-0" # Add margin-bottom only on medium/small screens (when stacked)
        ),
        # Image Display Column
        # Takes 8/12 width on large screens (lg), full width (12/12) on medium and smaller
        dbc.Col([
            # Row for Original and Enhanced Images
            dbc.Row([
                # Original Image Col
                # Takes 6/12 width (half) on large screens, full width on medium/small
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Original Image"),
                        dbc.CardBody(
                            dcc.Loading(dcc.Graph(id='original-image-graph', figure=create_plotly_fig(None, "Upload Image"), style={'height': '40vh'})), # Graph area with loading spinner
                            style={'padding': '5px', 'overflow': 'hidden'} # Card body styling
                        )
                    ]),
                    lg=6, md=12, # Responsive width
                    className="mb-3 mb-lg-0" # Add margin-bottom only on medium/small screens
                ),
                # Enhanced Image Col
                # Takes 6/12 width (half) on large screens, full width on medium/small
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Enhanced Image"),
                        dbc.CardBody(
                            dcc.Loading(dcc.Graph(id='enhanced-image-graph', figure=create_plotly_fig(None, "Apply Enhancement"), style={'height': '40vh'})),
                            style={'padding': '5px', 'overflow': 'hidden'}
                        )
                    ]),
                    lg=6, md=12 # Responsive width
                ),
            ]),
            # Row for Comparison View
            dbc.Row(
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Comparison View"),
                        dbc.CardBody(
                            dcc.Loading(dcc.Graph(id='comparison-graph', figure=create_plotly_fig(None, "Click Compare"), style={'height': '75vh'})), # Comparison graph area
                            style={'padding': '5px', 'overflow': 'hidden'}
                        )
                    ]),
                    width=12 # Always takes full width of its parent column
                ),
                className="mt-3" # Margin top for spacing from images above
            )
        ], lg=8, md=12) # Responsive width for the main image column
    ]),
    # Hidden Storage Components
    dcc.Store(id='store-original-image-data'),    # Stores original image base64
    dcc.Store(id='store-enhanced-image-data'),    # Stores enhanced image base64
    dcc.Store(id='store-original-filename'),      # Stores original filename
], fluid=True) # fluid=True makes the container span the full viewport width

# --- Callbacks ---

# Callback to update dynamic parameter input based on filter type selection
@app.callback(
    Output('parameter-input-div', 'children'),
    Input('filter-type', 'value')
)
def update_parameter_input(filter_type):
    """Dynamically generates input elements based on the selected filter."""
    if filter_type == 'smooth':
        # Only need cutoff slider for smoothing
        return html.Div([
            dbc.Label("Cutoff Frequency (D0): Lower = More Blur", html_for="cutoff-slider"),
            dcc.Slider(id='cutoff-slider', min=1, max=100, step=1, value=30, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom", "always_visible": True})
        ])
    elif filter_type == 'sharpen':
        # Only need cutoff slider for sharpening
        return html.Div([
            dbc.Label("Cutoff Frequency (D0): Lower = More Sharpening", html_for="cutoff-slider"),
            dcc.Slider(id='cutoff-slider', min=1, max=100, step=1, value=30, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom", "always_visible": True})
        ])
    elif filter_type == 'highboost':
        # Need cutoff slider AND boost factor input for high-boost
        return html.Div([
            dbc.Label("Cutoff Frequency (D0):", html_for="cutoff-slider"),
            dcc.Slider(id='cutoff-slider', min=1, max=100, step=1, value=30, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom", "always_visible": True}),
            dbc.Label("Boost Factor (A): > 1", html_for="boost-factor-input", className="mt-3"),
            dbc.Input(id='boost-factor-input', type='number', min=1.01, step=0.1, value=1.5) # Ensure min > 1
        ])
    # Return an empty Div if no filter type is matched (shouldn't normally happen with default value)
    return html.Div()

# Callback to handle image upload
@app.callback(
    Output('original-image-graph', 'figure'),
    Output('store-original-image-data', 'data'),
    Output('store-original-filename', 'data'),
    Output('status-message', 'children', allow_duplicate=True),
    Output('enhanced-image-graph', 'figure', allow_duplicate=True),
    Output('comparison-graph', 'figure', allow_duplicate=True),
    Output('compare-button', 'disabled', allow_duplicate=True),
    Output('store-enhanced-image-data', 'data', allow_duplicate=True), # Reset enhanced data on new upload
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call=True # Don't run on initial load
)
def handle_upload(contents, filename):
    """Processes uploaded image, displays it, and stores data."""
    print("--- handle_upload triggered ---")
    if contents is None:
        print("No contents, preventing update.")
        raise PreventUpdate # Prevent callback execution if no file uploaded

    # Define the expected outputs for error conditions to ensure tuple size matches
    outputs_on_error = (
        create_plotly_fig(None, "Upload Error"), None, None,
        "Error during upload processing.",
        create_plotly_fig(None, "Apply Enhancement"), create_plotly_fig(None, "Click Compare"),
        True, None
    )
    num_expected_outputs = 8 # Match the number of Output() lines

    try:
        print(f"Attempting to parse contents for filename: {filename}")
        img_orig_cv = parse_contents(contents) # Decode base64 to OpenCV image
        if img_orig_cv is None:
            print("Image parsing failed.")
            error_msg = "Error: Failed to decode image."
            # Modify the error message in the predefined error tuple
            outputs_on_error_list = list(outputs_on_error); outputs_on_error_list[3] = error_msg; return tuple(outputs_on_error_list)

        print(f"Successfully parsed image: {filename}, shape: {img_orig_cv.shape}")

        # Create Plotly figure for the original image
        print("Creating Plotly figure for original image...")
        fig_orig = create_plotly_fig(img_orig_cv, f"Original: {filename}")
        print(f"Original figure created. Type: {type(fig_orig)}. Has data: {len(fig_orig.data)>0}")

        # Encode the original OpenCV image back to base64 (PNG format) for storage
        print("Encoding original image to PNG base64...")
        _ret, buffer = cv2.imencode('.png', img_orig_cv)
        if not _ret: raise ValueError("Failed to encode image to PNG buffer.")
        img_orig_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"Base64 encoding successful (len > 0: {len(img_orig_base64)>0}).")

        # Define the outputs for a successful upload
        # Reset enhanced image graph, comparison graph, disable compare button, clear enhanced data store
        outputs_on_success = (
            fig_orig, img_orig_base64, filename,
            f"Loaded: {filename}",
            create_plotly_fig(None, "Apply Enhancement"), create_plotly_fig(None, "Click Compare"),
            True, None # Disable compare button, clear enhanced data
        )

        # Sanity check: ensure the number of returned items matches the Outputs
        if len(outputs_on_success) != num_expected_outputs:
            print(f"!!! DEV ERROR: Mismatch in success outputs ({num_expected_outputs}) vs actual ({len(outputs_on_success)}).");
            outputs_on_error_list = list(outputs_on_error); outputs_on_error_list[3] = "Internal Error: Output count mismatch."; return tuple(outputs_on_error_list)

        print(f"handle_upload returning successfully: {outputs_on_success[3]}")
        return outputs_on_success

    except Exception as e:
        print(f"!!! EXCEPTION in handle_upload: {type(e).__name__}: {e}"); traceback.print_exc();
        # Modify and return the predefined error tuple
        outputs_on_error_list = list(outputs_on_error); outputs_on_error_list[3] = f"Error processing upload: {e}"; return tuple(outputs_on_error_list)


# Callback to apply selected enhancement technique
@app.callback(
    Output('enhanced-image-graph', 'figure', allow_duplicate=True),
    Output('store-enhanced-image-data', 'data', allow_duplicate=True),
    Output('compare-button', 'disabled', allow_duplicate=True), # Enable compare button on success
    Output('status-message', 'children', allow_duplicate=True),
    Input('enhance-button', 'n_clicks'),
    State('store-original-image-data', 'data'), # Get original image data
    State('filter-type', 'value'),             # Get selected filter type
    State('cutoff-slider', 'value'),           # Get cutoff value (might exist or not)
    State('boost-factor-input', 'value'),      # Get boost factor (might exist or not)
    prevent_initial_call=True
)
def apply_enhancement(n_clicks, img_orig_base64, filter_type, cutoff, boost_factor): # boost_factor might be None if input doesn't exist
    """Applies the selected filter, displays result, and enables comparison."""
    if n_clicks is None: raise PreventUpdate # Should not trigger on load

    print(f"--- apply_enhancement triggered. Filter: {filter_type}, Cutoff: {cutoff}, Boost Input: {boost_factor} ---")

    # Define expected outputs for error case matching the callback definition
    num_outputs = 4 # Match the number of Output() lines
    outputs_on_error = (create_plotly_fig(None, "Enhancement Error"), None, True, "Error during enhancement.")

    # Check if original image data exists
    if not img_orig_base64:
        print("!!! apply_enhancement: Received EMPTY original base64 data!")
        error_outputs = list(outputs_on_error)
        error_outputs[3] = "Error: Original data missing. Please upload an image first."
        return tuple(error_outputs)

    print(f"--- apply_enhancement: Received original base64 (len={len(img_orig_base64)}, starts='{img_orig_base64[:20]}...') ---")
    img_orig_cv = None; img_enhanced_cv = None; status = ""; img_enhanced_base64 = None
    start_time = datetime.datetime.now()

    try:
        # Decode stored base64 original image
        decoded = base64.b64decode(img_orig_base64); buffer_np = np.frombuffer(decoded, dtype=np.uint8); img_orig_cv = cv2.imdecode(buffer_np, cv2.IMREAD_GRAYSCALE)
        if img_orig_cv is None: raise ValueError("cv2.imdecode failed for original image in apply_enhancement")
        print(f"--- apply_enhancement: Decoded original shape: {img_orig_cv.shape}, dtype: {img_orig_cv.dtype} ---")

        # --- Apply Filter based on type ---
        # Check if cutoff frequency is provided (needed for all filter types here)
        if cutoff is None:
             status = "Error: Cutoff Frequency (D0) is required."
             raise ValueError(status)

        if filter_type == 'smooth':
            print(f"Applying Smoothing: Cutoff={cutoff}")
            glpf_mask = create_gaussian_lowpass_filter(img_orig_cv.shape, cutoff)
            img_enhanced_cv = apply_frequency_filter(img_orig_cv, glpf_mask)
            status = f"Smoothed (C={cutoff})"

        elif filter_type == 'sharpen':
             print(f"Applying Sharpening: Cutoff={cutoff}")
             ghpf_mask = create_gaussian_highpass_filter(img_orig_cv.shape, cutoff)
             img_enhanced_cv = apply_frequency_filter(img_orig_cv, ghpf_mask)
             status = f"Sharpened (C={cutoff})"

        elif filter_type == 'highboost':
             # Check boost_factor ONLY if type is highboost
             if boost_factor is None: # Check if the input existed and had a value
                 status = "Error: Boost Factor (A) is required for High-Boost."
                 raise ValueError(status)
             try:
                 # Validate and convert boost_factor
                 current_boost_factor = float(boost_factor)
                 if current_boost_factor <= 1.0:
                      status = "Error: Boost Factor (A) must be greater than 1.0."
                      raise ValueError(status)
             except (ValueError, TypeError): # Catch if conversion to float fails
                 status = f"Error: Invalid Boost Factor value '{boost_factor}'. Must be a number > 1.0."
                 raise ValueError(status)

             # Proceed with high-boost filtering
             print(f"Applying High-Boost: C={cutoff}, B={current_boost_factor}")
             ghpf_mask = create_gaussian_highpass_filter(img_orig_cv.shape, cutoff)
             # High-boost mask = (A-1) + Highpass = (A-1) + (1 - Lowpass) = A - Lowpass? No, definition is (A-1)I + H(I) = I + (A-1)I + H(I) - I
             # Mask in frequency domain: A*F - LPF*F => (A-1)*F + HPF*F => Mask = A-1 + HPF_Mask = A-1 + (1 - LPF_Mask) = A - LPF_Mask
             # Alternative: Mask = 1 + (A-1) * HPF_Mask ... let's stick to the common definition using addition
             highboost_mask = (current_boost_factor - 1) + ghpf_mask
             img_enhanced_cv = apply_frequency_filter(img_orig_cv, highboost_mask)
             status = f"High-Boost (C={cutoff}, B={current_boost_factor:.1f})"

        else:
            # Handle case where filter_type is somehow invalid
            status = "Error: Unknown filter type selected."
            raise ValueError(status)

        # Check if filtering function returned a valid image
        if img_enhanced_cv is None:
            status = f"Error: Enhancement function returned None for {filter_type}."
            raise ValueError(status)
        print(f"--- apply_enhancement: Enhanced image generated. Shape: {img_enhanced_cv.shape}, dtype: {img_enhanced_cv.dtype} ---")

        # Encode the enhanced image to base64 (PNG) for storage and potential download
        print("--- apply_enhancement: Encoding enhanced image to PNG buffer... ---")
        _ret_enh, buffer_enh = cv2.imencode('.png', img_enhanced_cv);
        if not _ret_enh:
            status = "Error: cv2.imencode failed for enhanced image."
            raise ValueError(status)
        print(f"--- apply_enhancement: cv2.imencode success. Buffer length: {len(buffer_enh)} ---")
        img_enhanced_base64 = base64.b64encode(buffer_enh).decode('utf-8')
        if not img_enhanced_base64: # Check if encoding resulted in non-empty string
            status = "Error: Base64 encoding resulted in empty string."
            raise ValueError(status)
        print(f"--- apply_enhancement: Enhanced image base64 encoded (len={len(img_enhanced_base64)}, starts='{img_enhanced_base64[:20]}...') ---")

        # Create Plotly figure for the enhanced image
        fig_enhanced = create_plotly_fig(img_enhanced_cv, f"Enhanced: {status}")
        print(f"Enhanced figure created. Type: {type(fig_enhanced)}. Has data: {len(fig_enhanced.data)>0}")

        # Prepare success outputs
        end_time = datetime.datetime.now(); processing_time = (end_time - start_time).total_seconds()
        final_status = f"Applied: {status} ({processing_time:.2f}s)"
        print(f"apply_enhancement returning successfully: {final_status}")
        # Enable compare button on success
        outputs_on_success = (fig_enhanced, img_enhanced_base64, False, final_status)

        # Final sanity check on output tuple size
        if len(outputs_on_success) != num_outputs:
             print(f"!!! DEV ERROR: Mismatch in success outputs ({num_outputs}) vs actual ({len(outputs_on_success)}).")
             error_outputs = list(outputs_on_error)
             error_outputs[3] = "Internal Error: Output count mismatch on success."
             return tuple(error_outputs)
        return outputs_on_success

    except Exception as e:
        print(f"!!! EXCEPTION in apply_enhancement: {type(e).__name__}: {e}"); traceback.print_exc()
        # Use status message if set by validation, otherwise format the exception message
        error_status = status if status.startswith("Error:") else f"Error applying {filter_type}: {e}"
        # Prepare error outputs
        error_outputs = list(outputs_on_error)
        error_outputs[3] = error_status # Update the status message part

        # Final sanity check on error output tuple size
        if len(error_outputs) != num_outputs:
             print(f"!!! DEV ERROR: Mismatch in error outputs ({num_outputs}) vs actual ({len(error_outputs)}).")
             # Attempt to return a failsafe tuple matching the count
             safe_error_outputs = [create_plotly_fig(None, "Enhancement Error"), None, True, error_status]
             return tuple(safe_error_outputs[:num_outputs]) # Slice to ensure correct length
        return tuple(error_outputs)


# Callback to generate and display the comparison plot
@app.callback(
    Output('comparison-graph', 'figure', allow_duplicate=True),
    Output('status-message', 'children', allow_duplicate=True),
    Input('compare-button', 'n_clicks'),
    State('store-original-image-data', 'data'), # Get original base64
    State('store-enhanced-image-data', 'data'), # Get enhanced base64
    State('store-original-filename', 'data'),   # Get filename for context
    prevent_initial_call=True
)
def show_comparison(n_clicks, img_orig_base64, img_enhanced_base64, filename):
    """Generates the comparison figure when the compare button is clicked."""
    if n_clicks is None: raise PreventUpdate # Don't run if button hasn't been clicked

    print(f"--- show_comparison triggered ---")
    # Check if required image data is present
    if not img_orig_base64: print("!!! show_comparison: Received EMPTY original base64 data!"); return create_plotly_fig(None, "Comparison Error: Missing Original"), "Error: Missing original data. Please upload and enhance first."
    if not img_enhanced_base64: print("!!! show_comparison: Received EMPTY enhanced base64 data!"); return create_plotly_fig(None, "Comparison Error: Missing Enhanced"), "Error: Missing enhanced data. Please apply enhancement first."

    print(f"--- show_comparison: Received original base64 (len={len(img_orig_base64)}, starts='{img_orig_base64[:20]}...') ---")
    print(f"--- show_comparison: Received enhanced base64 (len={len(img_enhanced_base64)}, starts='{img_enhanced_base64[:20]}...') ---")
    img_orig_cv = None; img_enhanced_cv = None

    try:
        # Decode both base64 images back to OpenCV format
        print("--- show_comparison: Decoding original base64... ---"); decoded_orig = base64.b64decode(img_orig_base64); buffer_orig = np.frombuffer(decoded_orig, dtype=np.uint8); img_orig_cv = cv2.imdecode(buffer_orig, cv2.IMREAD_GRAYSCALE)
        if img_orig_cv is None: raise ValueError("Failed to decode stored original image for comparison.")
        print(f"--- show_comparison: Decoded original shape: {img_orig_cv.shape}, dtype: {img_orig_cv.dtype} ---")

        print("--- show_comparison: Decoding enhanced base64... ---"); decoded_enh = base64.b64decode(img_enhanced_base64); buffer_enh = np.frombuffer(decoded_enh, dtype=np.uint8); img_enhanced_cv = cv2.imdecode(buffer_enh, cv2.IMREAD_GRAYSCALE)
        if img_enhanced_cv is None: raise ValueError("Failed to decode stored enhanced image for comparison.")
        print(f"--- show_comparison: Decoded enhanced shape: {img_enhanced_cv.shape}, dtype: {img_enhanced_cv.dtype} ---")

        # Prepare a shorter filename for display if needed
        enh_filename_short = "Enhanced"; fname_base = "image"
        if filename: fname_base = os.path.splitext(os.path.basename(filename))[0]; enh_filename_short = f"{fname_base}_enh"
        if len(enh_filename_short) > 30: enh_filename_short = enh_filename_short[:15] + "..." + enh_filename_short[-12:] # Truncate long names

        # Generate the comparison figure using the helper function
        print("--- show_comparison: Generating comparison figure (1x2)... ---")
        start_time = datetime.datetime.now()
        comparison_fig = create_comparison_figure(img_orig_cv, img_enhanced_cv, enh_filename_short)
        print(f"Comparison figure created. Type: {type(comparison_fig)}. Has data: {len(comparison_fig.data)>0}")

        # Prepare success status message
        end_time = datetime.datetime.now(); processing_time = (end_time - start_time).total_seconds(); final_status = f"Comparison view generated ({processing_time:.2f}s)."
        print(f"show_comparison returning successfully: {final_status}")
        # Return the figure and status message
        return comparison_fig, final_status

    except Exception as e:
        print(f"!!! EXCEPTION in show_comparison: {type(e).__name__}: {e}"); traceback.print_exc()
        error_message = f"Error generating comparison: {e}"
        # Return an error figure and the error message
        return create_plotly_fig(None, "Comparison Error"), error_message


# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash server...")
    # Use debug=True for development (enables hot-reloading and error pages)
    # host='127.0.0.1' makes it accessible only locally
    # host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='127.0.0.1')