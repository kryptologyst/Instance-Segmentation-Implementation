#!/usr/bin/env python3
"""
Demo script showcasing the instance segmentation capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from instance_segmentation import InstanceSegmentationPipeline
from data_generator import SyntheticDataGenerator
from visualization import SegmentationVisualizer


def main():
    """Run the demo."""
    print("🔍 Instance Segmentation Demo")
    print("=" * 40)
    
    # Generate synthetic data
    print("🎨 Generating synthetic test image...")
    generator = SyntheticDataGenerator(image_size=(640, 480))
    image, annotations = generator.generate_image(num_objects=5)
    
    # Save the image
    from PIL import Image
    image_path = Path("demo_image.jpg")
    Image.fromarray(image).save(image_path)
    print(f"✅ Saved test image: {image_path}")
    
    # Initialize pipeline
    print("🤖 Initializing instance segmentation pipeline...")
    pipeline = InstanceSegmentationPipeline(
        model_name="maskrcnn_resnet50_fpn",
        confidence_threshold=0.3  # Lower threshold for demo
    )
    print("✅ Pipeline initialized")
    
    # Process the image
    print("🔍 Processing image...")
    predictions = pipeline.process_image(
        image_path,
        output_path="demo_result.jpg",
        show_results=False
    )
    
    # Display results
    print(f"\n📊 Detection Results:")
    print(f"Found {len(predictions['boxes'])} instances:")
    
    for i, (label, score, box) in enumerate(zip(
        predictions['labels'],
        predictions['scores'],
        predictions['boxes']
    )):
        print(f"  {i+1}. Class {label}, Confidence: {score:.3f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    visualizer = SegmentationVisualizer()
    
    # Load the segmented result
    if Path("demo_result.jpg").exists():
        segmented_image = Image.open("demo_result.jpg")
        segmented_array = np.array(segmented_image)
        
        # Create comparison plot
        visualizer.create_comparison_plot(
            image, segmented_array, predictions, annotations, "demo_comparison.png"
        )
        print("✅ Created comparison plot: demo_comparison.png")
        
        # Create detection summary
        visualizer.create_detection_summary(predictions, "demo_summary.png")
        print("✅ Created detection summary: demo_summary.png")
    
    print("\n🎉 Demo completed successfully!")
    print("\nGenerated files:")
    print("- demo_image.jpg (original synthetic image)")
    print("- demo_result.jpg (segmentation result)")
    print("- demo_comparison.png (comparison visualization)")
    print("- demo_summary.png (detection summary)")
    
    print("\nTo run the web interface:")
    print("streamlit run web_app/app.py")
    
    print("\nTo use the CLI:")
    print("python cli.py demo_image.jpg --show-results")


if __name__ == "__main__":
    import numpy as np
    main()
