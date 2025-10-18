"""
Instance Segmentation Implementation

This module provides a modern implementation of instance segmentation using
state-of-the-art models from PyTorch and Hugging Face Transformers.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstanceSegmentationModel:
    """
    A modern instance segmentation model wrapper supporting multiple architectures.
    """
    
    def __init__(
        self,
        model_name: str = "maskrcnn_resnet50_fpn",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> None:
        """
        Initialize the instance segmentation model.
        
        Args:
            model_name: Name of the model to use ('maskrcnn_resnet50_fpn', 'maskrcnn_resnet101_fpn', 'fasterrcnn_resnet50_fpn')
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection)
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing {model_name} on {self.device}")
        self._load_model()
        self._setup_transforms()
        
    def _load_model(self) -> None:
        """Load the specified model architecture."""
        try:
            if self.model_name == "maskrcnn_resnet50_fpn":
                self.model = models.detection.maskrcnn_resnet50_fpn(
                    weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
                )
            elif self.model_name == "maskrcnn_resnet101_fpn":
                self.model = models.detection.maskrcnn_resnet101_fpn(
                    weights=models.detection.MaskRCNN_ResNet101_FPN_Weights.COCO_V1
                )
            elif self.model_name == "fasterrcnn_resnet50_fpn":
                self.model = models.detection.fasterrcnn_resnet50_fpn(
                    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Perform instance segmentation inference.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary containing predictions (boxes, labels, scores, masks)
        """
        try:
            with torch.no_grad():
                predictions = self.model(image_tensor)[0]
            
            # Filter predictions by confidence threshold
            valid_indices = predictions['scores'] >= self.confidence_threshold
            
            filtered_predictions = {
                'boxes': predictions['boxes'][valid_indices].cpu().numpy(),
                'labels': predictions['labels'][valid_indices].cpu().numpy(),
                'scores': predictions['scores'][valid_indices].cpu().numpy(),
                'masks': predictions['masks'][valid_indices].cpu().numpy()
            }
            
            logger.info(f"Found {len(filtered_predictions['boxes'])} instances above threshold {self.confidence_threshold}")
            return filtered_predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise


class InstanceSegmentationVisualizer:
    """
    Handles visualization of instance segmentation results.
    """
    
    def __init__(self, colormap: str = "tab20") -> None:
        """
        Initialize the visualizer.
        
        Args:
            colormap: Matplotlib colormap name for instance colors
        """
        self.colormap = plt.cm.get_cmap(colormap)
        self.coco_labels = self._load_coco_labels()
    
    def _load_coco_labels(self) -> Dict[int, str]:
        """Load COCO dataset class labels."""
        return {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
            38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
            42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
            47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
            52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
            57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
            62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
            70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
            76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
            85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
            90: 'toothbrush'
        }
    
    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
        show_labels: bool = True,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Visualize instance segmentation predictions on the image.
        
        Args:
            image: Original input image
            predictions: Model predictions dictionary
            save_path: Optional path to save the visualization
            show_labels: Whether to show class labels
            alpha: Transparency for mask overlay
            
        Returns:
            Image with visualizations applied
        """
        output_image = image.copy()
        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']
        masks = predictions['masks']
        
        # Generate colors for each instance
        colors = self._generate_colors(len(boxes))
        
        for i in range(len(boxes)):
            # Get mask and make it binary
            mask = masks[i, 0]
            binary_mask = mask > 0.5
            
            # Create colored mask
            color = colors[i]
            colored_region = np.zeros_like(image)
            for c in range(3):  # RGB channels
                colored_region[:, :, c] = binary_mask * color[c]
            
            # Blend with original image
            output_image = cv2.addWeighted(output_image, 1.0, colored_region, alpha, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color.tolist(), 2)
            
            # Add label and confidence
            if show_labels:
                label_id = labels[i]
                label_name = self.coco_labels.get(label_id, f"class_{label_id}")
                confidence = scores[i]
                text = f"{label_name}: {confidence:.2f}"
                
                # Calculate text position
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
                
                cv2.putText(
                    output_image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2
                )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            logger.info(f"Visualization saved to {save_path}")
        
        return output_image
    
    def _generate_colors(self, num_instances: int) -> List[np.ndarray]:
        """Generate distinct colors for each instance."""
        colors = []
        for i in range(num_instances):
            color = self.colormap(i / max(1, num_instances - 1))[:3]
            color = np.array(color) * 255
            colors.append(color.astype(np.uint8))
        return colors
    
    def plot_results(
        self,
        original_image: np.ndarray,
        segmented_image: np.ndarray,
        predictions: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (15, 7)
    ) -> None:
        """
        Create a side-by-side comparison plot.
        
        Args:
            original_image: Original input image
            segmented_image: Image with segmentation overlays
            predictions: Model predictions
            figsize: Figure size for the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(segmented_image)
        ax2.set_title(f'Instance Segmentation ({len(predictions["boxes"])} instances)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()


class InstanceSegmentationPipeline:
    """
    Complete pipeline for instance segmentation with modern features.
    """
    
    def __init__(
        self,
        model_name: str = "maskrcnn_resnet50_fpn",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> None:
        """
        Initialize the complete pipeline.
        
        Args:
            model_name: Model architecture to use
            device: Device for inference
            confidence_threshold: Minimum confidence for detections
        """
        self.model = InstanceSegmentationModel(model_name, device, confidence_threshold)
        self.visualizer = InstanceSegmentationVisualizer()
    
    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_results: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save results
            show_results: Whether to display results
            
        Returns:
            Dictionary containing predictions
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load and preprocess image
        image_tensor = self.model.preprocess_image(image_path)
        original_image = np.array(Image.open(image_path).convert("RGB"))
        
        # Run inference
        predictions = self.model.predict(image_tensor)
        
        # Visualize results
        segmented_image = self.visualizer.visualize_predictions(
            original_image, predictions, output_path
        )
        
        # Show results
        if show_results:
            self.visualizer.plot_results(original_image, segmented_image, predictions)
        
        return predictions
    
    def batch_process(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Optional directory to save results
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        output_dir = Path(output_dir) if output_dir else None
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            output_path = None
            if output_dir:
                output_path = output_dir / f"result_{i+1:03d}.jpg"
            
            try:
                predictions = self.process_image(
                    image_path, output_path, show_results=False
                )
                results.append(predictions)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({})
        
        return results


def main() -> None:
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Instance Segmentation Pipeline")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", 
                       choices=["maskrcnn_resnet50_fpn", "maskrcnn_resnet101_fpn", "fasterrcnn_resnet50_fpn"],
                       help="Model architecture to use")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Confidence threshold for detections")
    parser.add_argument("--output", help="Output path for results")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InstanceSegmentationPipeline(
        model_name=args.model,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    # Process image
    predictions = pipeline.process_image(args.image_path, args.output)
    
    print(f"Found {len(predictions['boxes'])} instances")
    for i, (label, score) in enumerate(zip(predictions['labels'], predictions['scores'])):
        print(f"Instance {i+1}: Class {label}, Confidence {score:.3f}")


if __name__ == "__main__":
    main()
