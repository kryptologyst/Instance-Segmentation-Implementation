"""
Visualization utilities for instance segmentation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

logger = logging.getLogger(__name__)


class SegmentationVisualizer:
    """
    Advanced visualization tools for instance segmentation results.
    """
    
    def __init__(self, style: str = "default") -> None:
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        plt.style.use(style)
        
        # Set up color palette
        self.colors = sns.color_palette("husl", 20)
        
        # COCO class names
        self.coco_labels = {
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
    
    def create_detection_summary(
        self,
        predictions: Dict[str, np.ndarray],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create a summary visualization of detection results.
        
        Args:
            predictions: Model predictions
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        if len(predictions['labels']) > 0:
            unique_labels, counts = np.unique(predictions['labels'], return_counts=True)
            class_names = [self.coco_labels.get(label, f"Class {label}") for label in unique_labels]
            
            ax1.bar(class_names, counts)
            ax1.set_title('Object Class Distribution')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Confidence distribution
            ax2.hist(predictions['scores'], bins=20, alpha=0.7, edgecolor='black')
            ax2.set_title('Confidence Score Distribution')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(predictions['scores']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(predictions['scores']):.3f}')
            ax2.legend()
            
            # Box area distribution
            boxes = predictions['boxes']
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ax3.hist(areas, bins=20, alpha=0.7, edgecolor='black')
            ax3.set_title('Bounding Box Area Distribution')
            ax3.set_xlabel('Area (pixels²)')
            ax3.set_ylabel('Frequency')
            
            # Confidence vs Area scatter
            ax4.scatter(areas, predictions['scores'], alpha=0.6)
            ax4.set_title('Confidence vs Box Area')
            ax4.set_xlabel('Area (pixels²)')
            ax4.set_ylabel('Confidence Score')
            
            # Add correlation coefficient
            corr = np.corrcoef(areas, predictions['scores'])[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            # No detections
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title('No Data Available')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detection summary saved to {save_path}")
        
        return fig
    
    def create_comparison_plot(
        self,
        original_image: np.ndarray,
        segmented_image: np.ndarray,
        predictions: Dict[str, np.ndarray],
        ground_truth: Optional[List[Dict]] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create a comparison plot showing original, segmented, and analysis.
        
        Args:
            original_image: Original input image
            segmented_image: Image with segmentation overlays
            predictions: Model predictions
            ground_truth: Optional ground truth annotations
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original_image)
        ax1.set_title('Original Image', fontsize=14)
        ax1.axis('off')
        
        # Segmented image
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(segmented_image)
        ax2.set_title(f'Segmentation Result ({len(predictions["boxes"])} instances)', fontsize=14)
        ax2.axis('off')
        
        # Detection details
        ax3 = plt.subplot(2, 3, 3)
        if len(predictions['boxes']) > 0:
            # Create detection table
            detection_data = []
            for i, (label, score, box) in enumerate(zip(
                predictions['labels'], predictions['scores'], predictions['boxes']
            )):
                class_name = self.coco_labels.get(label, f"Class {label}")
                area = (box[2] - box[0]) * (box[3] - box[1])
                detection_data.append({
                    'Instance': i + 1,
                    'Class': class_name,
                    'Confidence': f"{score:.3f}",
                    'Area': f"{area:.0f}"
                })
            
            # Create table
            table_data = [[d['Instance'], d['Class'], d['Confidence'], d['Area']] 
                         for d in detection_data]
            table = ax3.table(cellText=table_data,
                             colLabels=['#', 'Class', 'Confidence', 'Area'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax3.axis('off')
            ax3.set_title('Detection Details', fontsize=14)
        else:
            ax3.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Detection Details', fontsize=14)
        
        # Ground truth comparison (if available)
        ax4 = plt.subplot(2, 3, 4)
        if ground_truth:
            gt_classes = [ann['class'] for ann in ground_truth]
            unique_gt_classes, gt_counts = np.unique(gt_classes, return_counts=True)
            
            pred_classes = [self.coco_labels.get(label, f"Class {label}") 
                           for label in predictions['labels']]
            unique_pred_classes, pred_counts = np.unique(pred_classes, return_counts=True)
            
            # Create comparison bar chart
            all_classes = list(set(unique_gt_classes) | set(unique_pred_classes))
            gt_values = [gt_counts[list(unique_gt_classes).index(c)] if c in unique_gt_classes else 0 
                        for c in all_classes]
            pred_values = [pred_counts[list(unique_pred_classes).index(c)] if c in unique_pred_classes else 0 
                          for c in all_classes]
            
            x = np.arange(len(all_classes))
            width = 0.35
            
            ax4.bar(x - width/2, gt_values, width, label='Ground Truth', alpha=0.8)
            ax4.bar(x + width/2, pred_values, width, label='Predictions', alpha=0.8)
            
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Count')
            ax4.set_title('Ground Truth vs Predictions')
            ax4.set_xticks(x)
            ax4.set_xticklabels(all_classes, rotation=45)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No ground truth available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Ground Truth Comparison', fontsize=14)
        
        # Performance metrics
        ax5 = plt.subplot(2, 3, 5)
        if len(predictions['boxes']) > 0:
            metrics = {
                'Total Instances': len(predictions['boxes']),
                'Avg Confidence': np.mean(predictions['scores']),
                'Max Confidence': np.max(predictions['scores']),
                'Min Confidence': np.min(predictions['scores']),
                'Unique Classes': len(np.unique(predictions['labels']))
            }
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax5.bar(metric_names, metric_values)
            ax5.set_title('Performance Metrics')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}' if isinstance(value, float) else str(value),
                        ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'No metrics available', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Performance Metrics', fontsize=14)
        
        # Confidence distribution
        ax6 = plt.subplot(2, 3, 6)
        if len(predictions['scores']) > 0:
            ax6.hist(predictions['scores'], bins=10, alpha=0.7, edgecolor='black')
            ax6.set_title('Confidence Distribution')
            ax6.set_xlabel('Confidence Score')
            ax6.set_ylabel('Frequency')
            ax6.axvline(np.mean(predictions['scores']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(predictions['scores']):.3f}')
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'No confidence data', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Confidence Distribution', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def create_batch_analysis(
        self,
        results: List[Dict[str, np.ndarray]],
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create analysis plot for batch processing results.
        
        Args:
            results: List of prediction dictionaries
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data from all results
        all_scores = []
        all_labels = []
        instance_counts = []
        
        for result in results:
            if len(result['boxes']) > 0:
                all_scores.extend(result['scores'])
                all_labels.extend(result['labels'])
                instance_counts.append(len(result['boxes']))
            else:
                instance_counts.append(0)
        
        # Instance count per image
        ax1.hist(instance_counts, bins=max(1, len(set(instance_counts))), 
                alpha=0.7, edgecolor='black')
        ax1.set_title('Instance Count Distribution')
        ax1.set_xlabel('Number of Instances per Image')
        ax1.set_ylabel('Number of Images')
        
        # Overall confidence distribution
        if all_scores:
            ax2.hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_title('Overall Confidence Distribution')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(all_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_scores):.3f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Overall Confidence Distribution', fontsize=14)
        
        # Class distribution across all images
        if all_labels:
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            class_names = [self.coco_labels.get(label, f"Class {label}") for label in unique_labels]
            
            ax3.bar(class_names, counts)
            ax3.set_title('Overall Class Distribution')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Total Count')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Overall Class Distribution', fontsize=14)
        
        # Processing statistics
        total_images = len(results)
        images_with_detections = sum(1 for result in results if len(result['boxes']) > 0)
        total_instances = sum(len(result['boxes']) for result in results)
        
        stats = {
            'Total Images': total_images,
            'Images with Detections': images_with_detections,
            'Total Instances': total_instances,
            'Avg Instances/Image': total_instances / total_images if total_images > 0 else 0
        }
        
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
        
        bars = ax4.bar(stat_names, stat_values)
        ax4.set_title('Processing Statistics')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, stat_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1f}' if isinstance(value, float) else str(value),
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Batch analysis saved to {save_path}")
        
        return fig


def main():
    """Demo function for visualization utilities."""
    import sys
    sys.path.append(str(Path(__file__).parent / "src"))
    
    from data_generator import SyntheticDataGenerator
    from instance_segmentation import InstanceSegmentationPipeline
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    image, annotations = generator.generate_image(num_objects=5)
    
    # Process with pipeline
    pipeline = InstanceSegmentationPipeline(confidence_threshold=0.1)
    
    # Save temporary image
    temp_path = Path("temp_demo.jpg")
    Image.fromarray(image).save(temp_path)
    
    try:
        predictions = pipeline.process_image(temp_path, show_results=False)
        
        # Create visualizations
        visualizer = SegmentationVisualizer()
        
        # Detection summary
        visualizer.create_detection_summary(predictions, "demo_summary.png")
        
        # Comparison plot
        segmented_image = np.array(Image.open("temp_output.jpg")) if Path("temp_output.jpg").exists() else image
        visualizer.create_comparison_plot(
            image, segmented_image, predictions, annotations, "demo_comparison.png"
        )
        
        print("Demo visualizations created: demo_summary.png, demo_comparison.png")
        
    finally:
        # Clean up
        temp_path.unlink(missing_ok=True)
        Path("temp_output.jpg").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
