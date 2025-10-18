#!/usr/bin/env python3
"""
Command-line interface for instance segmentation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from instance_segmentation import InstanceSegmentationPipeline
from advanced_segmentation import AdvancedInstanceSegmentationModel, ModelEnsemble
from data_generator import SyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Instance Segmentation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py image.jpg
  
  # With custom model and confidence
  python cli.py image.jpg --model maskrcnn_resnet101_fpn --confidence 0.8
  
  # Generate synthetic data
  python cli.py --generate-synthetic --num-images 10 --output-dir data/synthetic
  
  # Batch process directory
  python cli.py --batch-process data/images --output-dir results
  
  # Advanced features
  python cli.py image.jpg --use-nms --use-tta --device cuda
        """
    )
    
    # Input options
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to input image or directory"
    )
    
    # Model options
    parser.add_argument(
        "--model",
        choices=["maskrcnn_resnet50_fpn", "maskrcnn_resnet101_fpn", "fasterrcnn_resnet50_fpn"],
        default="maskrcnn_resnet50_fpn",
        help="Model architecture to use"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for inference"
    )
    
    # Advanced options
    parser.add_argument(
        "--use-nms",
        action="store_true",
        help="Use Non-Maximum Suppression"
    )
    
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Use Test Time Augmentation"
    )
    
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="NMS threshold"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for results"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch processing"
    )
    
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Display results using matplotlib"
    )
    
    # Data generation options
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic data"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of synthetic images to generate"
    )
    
    parser.add_argument(
        "--image-size",
        default="640x480",
        help="Size of synthetic images (WIDTHxHEIGHT)"
    )
    
    # Batch processing
    parser.add_argument(
        "--batch-process",
        help="Process all images in directory"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle device selection
    device = None if args.device == "auto" else args.device
    
    try:
        # Generate synthetic data
        if args.generate_synthetic:
            logger.info("Generating synthetic data...")
            
            # Parse image size
            width, height = map(int, args.image_size.split('x'))
            
            generator = SyntheticDataGenerator(image_size=(width, height))
            output_dir = args.output_dir or "data/synthetic"
            
            generator.generate_dataset(
                num_images=args.num_images,
                output_dir=output_dir
            )
            
            logger.info(f"Generated {args.num_images} synthetic images in {output_dir}")
            return
        
        # Batch processing
        if args.batch_process:
            logger.info(f"Batch processing images in {args.batch_process}")
            
            input_dir = Path(args.batch_process)
            if not input_dir.exists():
                logger.error(f"Input directory {input_dir} does not exist")
                return
            
            # Find image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [
                f for f in input_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            if not image_files:
                logger.error(f"No image files found in {input_dir}")
                return
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # Initialize pipeline
            if args.use_nms or args.use_tta:
                model = AdvancedInstanceSegmentationModel(
                    model_name=args.model,
                    device=device,
                    confidence_threshold=args.confidence,
                    use_nms=args.use_nms,
                    nms_threshold=args.nms_threshold,
                    use_tta=args.use_tta
                )
                pipeline = InstanceSegmentationPipeline()
                pipeline.model = model
            else:
                pipeline = InstanceSegmentationPipeline(
                    model_name=args.model,
                    device=device,
                    confidence_threshold=args.confidence
                )
            
            # Process images
            results = pipeline.batch_process(
                image_files,
                output_dir=args.output_dir
            )
            
            logger.info(f"Processed {len(results)} images")
            return
        
        # Single image processing
        if not args.input_path:
            parser.error("Input path is required for single image processing")
        
        input_path = Path(args.input_path)
        if not input_path.exists():
            logger.error(f"Input file {input_path} does not exist")
            return
        
        logger.info(f"Processing image: {input_path}")
        
        # Initialize pipeline
        if args.use_nms or args.use_tta:
            model = AdvancedInstanceSegmentationModel(
                model_name=args.model,
                device=device,
                confidence_threshold=args.confidence,
                use_nms=args.use_nms,
                nms_threshold=args.nms_threshold,
                use_tta=args.use_tta
            )
            pipeline = InstanceSegmentationPipeline()
            pipeline.model = model
        else:
            pipeline = InstanceSegmentationPipeline(
                model_name=args.model,
                device=device,
                confidence_threshold=args.confidence
            )
        
        # Process image
        predictions = pipeline.process_image(
            input_path,
            output_path=args.output,
            show_results=args.show_results
        )
        
        # Print results
        print(f"\nDetection Results:")
        print(f"Model: {args.model}")
        print(f"Confidence Threshold: {args.confidence}")
        print(f"Device: {device or 'auto-detect'}")
        print(f"Found {len(predictions['boxes'])} instances:")
        
        for i, (label, score, box) in enumerate(zip(
            predictions['labels'],
            predictions['scores'],
            predictions['boxes']
        )):
            print(f"  {i+1}. Class {label}, Confidence: {score:.3f}, "
                  f"Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
        
        if args.output:
            print(f"\nResults saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
