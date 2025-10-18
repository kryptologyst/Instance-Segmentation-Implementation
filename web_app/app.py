"""
Streamlit web interface for instance segmentation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from instance_segmentation import InstanceSegmentationPipeline
from data_generator import SyntheticDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Instance Segmentation Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_pipeline(model_name: str, confidence_threshold: float, device: Optional[str]) -> InstanceSegmentationPipeline:
    """Load the instance segmentation pipeline."""
    try:
        return InstanceSegmentationPipeline(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold
        )
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Instance Segmentation Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases modern instance segmentation using state-of-the-art models.
    Upload an image or use our synthetic data generator to see instance segmentation in action!
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Model Architecture",
        ["maskrcnn_resnet50_fpn", "maskrcnn_resnet101_fpn", "fasterrcnn_resnet50_fpn"],
        help="Choose the model architecture for instance segmentation"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Device selection
    device_options = ["Auto-detect", "CPU", "CUDA"]
    device_choice = st.sidebar.selectbox("Device", device_options)
    device = None if device_choice == "Auto-detect" else device_choice.lower()
    
    # Visualization options
    st.sidebar.header("🎨 Visualization")
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    alpha = st.sidebar.slider("Mask Transparency", 0.1, 1.0, 0.5, 0.1)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Upload Image", "🎲 Generate Synthetic", "📊 Model Info"])
    
    with tab1:
        st.header("Upload Your Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to perform instance segmentation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = Path("temp_upload.jpg")
                        image.save(temp_path)
                        
                        # Load pipeline
                        pipeline = load_pipeline(model_name, confidence_threshold, device)
                        
                        if pipeline is not None:
                            # Process image
                            predictions = pipeline.process_image(
                                temp_path, 
                                show_results=False
                            )
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📊 Detection Results")
                                
                                if len(predictions['boxes']) > 0:
                                    # Create results dataframe
                                    import pandas as pd
                                    
                                    results_data = []
                                    for i, (label, score, box) in enumerate(zip(
                                        predictions['labels'], 
                                        predictions['scores'], 
                                        predictions['boxes']
                                    )):
                                        results_data.append({
                                            "Instance": i + 1,
                                            "Class ID": int(label),
                                            "Confidence": f"{score:.3f}",
                                            "Bounding Box": f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
                                        })
                                    
                                    results_df = pd.DataFrame(results_data)
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Metrics
                                    col1_1, col1_2, col1_3 = st.columns(3)
                                    with col1_1:
                                        st.metric("Total Instances", len(predictions['boxes']))
                                    with col1_2:
                                        avg_confidence = np.mean(predictions['scores'])
                                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                                    with col1_3:
                                        max_confidence = np.max(predictions['scores'])
                                        st.metric("Max Confidence", f"{max_confidence:.3f}")
                                else:
                                    st.warning("No objects detected above the confidence threshold.")
                            
                            with col2:
                                st.subheader("🎨 Segmented Image")
                                
                                # Load the segmented result
                                output_path = Path("temp_output.jpg")
                                if output_path.exists():
                                    segmented_image = Image.open(output_path)
                                    st.image(segmented_image, caption="Segmentation Result", use_column_width=True)
                                    
                                    # Download button
                                    with open(output_path, "rb") as file:
                                        st.download_button(
                                            label="📥 Download Result",
                                            data=file.read(),
                                            file_name="segmentation_result.jpg",
                                            mime="image/jpeg"
                                        )
                                
                                # Clean up temp files
                                temp_path.unlink(missing_ok=True)
                                output_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        logger.error(f"Processing error: {e}")
    
    with tab2:
        st.header("Generate Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎲 Data Generator")
            
            num_objects = st.slider("Number of Objects", 2, 10, 5)
            image_size = st.selectbox("Image Size", ["640x480", "800x600", "1024x768"])
            background_color = st.color_picker("Background Color", "#f0f0f0")
            
            if st.button("🎨 Generate Synthetic Image", type="primary"):
                with st.spinner("Generating synthetic image..."):
                    try:
                        # Parse image size
                        width, height = map(int, image_size.split('x'))
                        
                        # Generate synthetic image
                        generator = SyntheticDataGenerator(image_size=(width, height))
                        image_array, annotations = generator.generate_image(
                            num_objects=num_objects,
                            background_color=tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))
                        )
                        
                        # Convert to PIL Image
                        synthetic_image = Image.fromarray(image_array)
                        
                        # Store in session state
                        st.session_state.synthetic_image = synthetic_image
                        st.session_state.synthetic_annotations = annotations
                        
                        st.success(f"Generated synthetic image with {len(annotations)} objects!")
                        
                    except Exception as e:
                        st.error(f"Error generating synthetic image: {e}")
        
        with col2:
            st.subheader("📊 Generated Image")
            
            if 'synthetic_image' in st.session_state:
                st.image(st.session_state.synthetic_image, caption="Synthetic Image", use_column_width=True)
                
                # Show annotations
                if 'synthetic_annotations' in st.session_state:
                    annotations = st.session_state.synthetic_annotations
                    
                    st.write("**Object Annotations:**")
                    for i, ann in enumerate(annotations):
                        st.write(f"• {ann['class']}: confidence {ann['confidence']:.3f}")
                
                # Process synthetic image
                if st.button("🔍 Analyze Synthetic Image", type="primary"):
                    with st.spinner("Processing synthetic image..."):
                        try:
                            # Save synthetic image temporarily
                            temp_path = Path("temp_synthetic.jpg")
                            st.session_state.synthetic_image.save(temp_path)
                            
                            # Load pipeline
                            pipeline = load_pipeline(model_name, confidence_threshold, device)
                            
                            if pipeline is not None:
                                # Process image
                                predictions = pipeline.process_image(
                                    temp_path,
                                    show_results=False
                                )
                                
                                # Display comparison
                                st.subheader("🎯 Detection vs Ground Truth")
                                
                                col2_1, col2_2 = st.columns(2)
                                
                                with col2_1:
                                    st.write("**Ground Truth:**")
                                    for ann in annotations:
                                        st.write(f"• {ann['class']}")
                                
                                with col2_2:
                                    st.write("**Detected:**")
                                    if len(predictions['boxes']) > 0:
                                        for label, score in zip(predictions['labels'], predictions['scores']):
                                            st.write(f"• Class {label} (confidence: {score:.3f})")
                                    else:
                                        st.write("No objects detected")
                                
                                # Clean up
                                temp_path.unlink(missing_ok=True)
                        
                        except Exception as e:
                            st.error(f"Error processing synthetic image: {e}")
    
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Model Architecture")
            
            model_info = {
                "maskrcnn_resnet50_fpn": {
                    "description": "Mask R-CNN with ResNet-50 backbone",
                    "parameters": "~44M",
                    "speed": "Fast",
                    "accuracy": "Good"
                },
                "maskrcnn_resnet101_fpn": {
                    "description": "Mask R-CNN with ResNet-101 backbone",
                    "parameters": "~58M",
                    "speed": "Medium",
                    "accuracy": "Better"
                },
                "fasterrcnn_resnet50_fpn": {
                    "description": "Faster R-CNN with ResNet-50 backbone",
                    "parameters": "~41M",
                    "speed": "Fast",
                    "accuracy": "Good (no masks)"
                }
            }
            
            current_model = model_info[model_name]
            
            st.write(f"**{model_name}**")
            st.write(f"Description: {current_model['description']}")
            st.write(f"Parameters: {current_model['parameters']}")
            st.write(f"Speed: {current_model['speed']}")
            st.write(f"Accuracy: {current_model['accuracy']}")
        
        with col2:
            st.subheader("📈 Performance Metrics")
            
            # Mock performance data
            import pandas as pd
            
            metrics_data = {
                "Model": ["Mask R-CNN R50", "Mask R-CNN R101", "Faster R-CNN R50"],
                "mAP": [37.9, 40.2, 37.4],
                "Speed (FPS)": [12.5, 9.8, 15.2],
                "Memory (GB)": [2.1, 2.8, 1.9]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
        
        st.subheader("🎯 COCO Classes")
        
        # Show COCO class information
        coco_classes = {
            1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
            6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
            11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
            16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
            21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe"
        }
        
        # Display classes in columns
        cols = st.columns(3)
        for i, (class_id, class_name) in enumerate(coco_classes.items()):
            with cols[i % 3]:
                st.write(f"{class_id}: {class_name}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Instance Segmentation Demo | Built with Streamlit & PyTorch
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
