"""
Enhanced instance segmentation with state-of-the-art techniques.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class AdvancedInstanceSegmentationModel:
    """
    Advanced instance segmentation model with state-of-the-art techniques.
    """
    
    def __init__(
        self,
        model_name: str = "maskrcnn_resnet50_fpn",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
        use_nms: bool = True,
        nms_threshold: float = 0.5,
        use_tta: bool = False  # Test Time Augmentation
    ) -> None:
        """
        Initialize the advanced instance segmentation model.
        
        Args:
            model_name: Model architecture to use
            device: Device for inference
            confidence_threshold: Minimum confidence for detections
            use_nms: Whether to use Non-Maximum Suppression
            nms_threshold: NMS threshold
            use_tta: Whether to use Test Time Augmentation
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_nms = use_nms
        self.nms_threshold = nms_threshold
        self.use_tta = use_tta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing advanced {model_name} on {self.device}")
        self._load_model()
        self._setup_transforms()
        
    def _load_model(self) -> None:
        """Load the specified model with advanced configurations."""
        try:
            if self.model_name == "maskrcnn_resnet50_fpn":
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
                )
            elif self.model_name == "maskrcnn_resnet101_fpn":
                self.model = torchvision.models.detection.maskrcnn_resnet101_fpn(
                    weights=torchvision.models.detection.MaskRCNN_ResNet101_FPN_Weights.COCO_V1
                )
            elif self.model_name == "fasterrcnn_resnet50_fpn":
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Advanced model configurations
            self.model.eval()
            self.model.to(self.device)
            
            # Enable mixed precision if available
            if hasattr(torch.cuda, 'amp') and self.device == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
                self.use_amp = True
            else:
                self.use_amp = False
            
            logger.info(f"Successfully loaded {self.model_name} with advanced features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_transforms(self) -> None:
        """Setup advanced image preprocessing transforms."""
        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Test Time Augmentation transforms
        self.tta_transforms = [
            transforms.Compose([transforms.ToTensor()]),
            transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=1.0)]),
            transforms.Compose([transforms.ToTensor(), transforms.RandomVerticalFlip(p=1.0)]),
            transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=90)]),
        ]
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess an image with advanced techniques.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.use_tta:
                # Use multiple augmented versions
                augmented_images = []
                for transform in self.tta_transforms:
                    aug_image = transform(image).unsqueeze(0)
                    augmented_images.append(aug_image)
                return torch.cat(augmented_images, dim=0).to(self.device)
            else:
                image_tensor = self.base_transform(image).unsqueeze(0)
                return image_tensor.to(self.device)
                
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Perform advanced instance segmentation inference.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary containing predictions
        """
        try:
            with torch.no_grad():
                if self.use_amp and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        predictions = self.model(image_tensor)
                else:
                    predictions = self.model(image_tensor)
            
            # Handle TTA predictions
            if self.use_tta and len(predictions) > 1:
                predictions = self._ensemble_tta_predictions(predictions)
            else:
                predictions = predictions[0]
            
            # Apply Non-Maximum Suppression
            if self.use_nms:
                predictions = self._apply_nms(predictions)
            
            # Filter by confidence threshold
            valid_indices = predictions['scores'] >= self.confidence_threshold
            
            filtered_predictions = {
                'boxes': predictions['boxes'][valid_indices].cpu().numpy(),
                'labels': predictions['labels'][valid_indices].cpu().numpy(),
                'scores': predictions['scores'][valid_indices].cpu().numpy(),
                'masks': predictions['masks'][valid_indices].cpu().numpy() if 'masks' in predictions else None
            }
            
            logger.info(f"Found {len(filtered_predictions['boxes'])} instances above threshold {self.confidence_threshold}")
            return filtered_predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _ensemble_tta_predictions(self, predictions_list: List[Dict]) -> Dict:
        """
        Ensemble predictions from Test Time Augmentation.
        
        Args:
            predictions_list: List of predictions from different augmentations
            
        Returns:
            Ensembled predictions
        """
        # Simple averaging approach - can be enhanced with more sophisticated methods
        all_boxes = []
        all_labels = []
        all_scores = []
        all_masks = []
        
        for pred in predictions_list:
            all_boxes.append(pred['boxes'])
            all_labels.append(pred['labels'])
            all_scores.append(pred['scores'])
            if 'masks' in pred:
                all_masks.append(pred['masks'])
        
        # Concatenate all predictions
        ensembled = {
            'boxes': torch.cat(all_boxes, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'scores': torch.cat(all_scores, dim=0)
        }
        
        if all_masks:
            ensembled['masks'] = torch.cat(all_masks, dim=0)
        
        return ensembled
    
    def _apply_nms(self, predictions: Dict) -> Dict:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            NMS-filtered predictions
        """
        try:
            # Apply NMS per class
            keep_indices = []
            
            for class_id in torch.unique(predictions['labels']):
                class_mask = predictions['labels'] == class_id
                class_boxes = predictions['boxes'][class_mask]
                class_scores = predictions['scores'][class_mask]
                
                if len(class_boxes) > 0:
                    # Apply NMS
                    keep = torchvision.ops.nms(class_boxes, class_scores, self.nms_threshold)
                    
                    # Get original indices
                    original_indices = torch.where(class_mask)[0][keep]
                    keep_indices.append(original_indices)
            
            if keep_indices:
                keep_indices = torch.cat(keep_indices)
                
                # Filter predictions
                filtered_predictions = {
                    'boxes': predictions['boxes'][keep_indices],
                    'labels': predictions['labels'][keep_indices],
                    'scores': predictions['scores'][keep_indices]
                }
                
                if 'masks' in predictions:
                    filtered_predictions['masks'] = predictions['masks'][keep_indices]
                
                return filtered_predictions
            
            return predictions
            
        except Exception as e:
            logger.warning(f"NMS failed: {e}")
            return predictions


class HuggingFaceSegmentationModel:
    """
    Integration with Hugging Face Transformers for instance segmentation.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50-panoptic",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> None:
        """
        Initialize Hugging Face model.
        
        Args:
            model_name: Hugging Face model name
            device: Device for inference
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading Hugging Face model: {model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Hugging Face model."""
        try:
            self.pipeline = hf_pipeline(
                "image-segmentation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise
    
    def predict(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Perform inference using Hugging Face model.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Predictions dictionary
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Run inference
            results = self.pipeline(image)
            
            # Convert to our format
            boxes = []
            labels = []
            scores = []
            masks = []
            
            for result in results:
                if result['score'] >= self.confidence_threshold:
                    # Extract bounding box from mask
                    mask = np.array(result['mask'])
                    y_indices, x_indices = np.where(mask > 0)
                    
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, x2 = x_indices.min(), x_indices.max()
                        y1, y2 = y_indices.min(), y_indices.max()
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(0)  # Generic label for HF models
                        scores.append(result['score'])
                        masks.append(mask)
            
            return {
                'boxes': np.array(boxes) if boxes else np.array([]).reshape(0, 4),
                'labels': np.array(labels) if labels else np.array([]),
                'scores': np.array(scores) if scores else np.array([]),
                'masks': np.array(masks) if masks else np.array([]).reshape(0, 1, 1, 1)
            }
            
        except Exception as e:
            logger.error(f"Hugging Face inference failed: {e}")
            raise


class ModelEnsemble:
    """
    Ensemble multiple models for improved performance.
    """
    
    def __init__(
        self,
        models: List[Union[AdvancedInstanceSegmentationModel, HuggingFaceSegmentationModel]],
        weights: Optional[List[float]] = None
    ) -> None:
        """
        Initialize model ensemble.
        
        Args:
            models: List of models to ensemble
            weights: Weights for each model (if None, equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def predict(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Perform ensemble prediction.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Ensembled predictions
        """
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                if isinstance(model, HuggingFaceSegmentationModel):
                    pred = model.predict(image_path)
                else:
                    image_tensor = model.preprocess_image(image_path)
                    pred = model.predict(image_tensor)
                all_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if not all_predictions:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'labels': np.array([]),
                'scores': np.array([]),
                'masks': np.array([]).reshape(0, 1, 1, 1)
            }
        
        # Ensemble predictions
        return self._ensemble_predictions(all_predictions)
    
    def _ensemble_predictions(self, predictions_list: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Ensemble multiple predictions.
        
        Args:
            predictions_list: List of prediction dictionaries
            
        Returns:
            Ensembled predictions
        """
        # Simple weighted averaging approach
        all_boxes = []
        all_labels = []
        all_scores = []
        all_masks = []
        
        for i, pred in enumerate(predictions_list):
            weight = self.weights[i]
            
            if len(pred['boxes']) > 0:
                all_boxes.append(pred['boxes'] * weight)
                all_labels.append(pred['labels'])
                all_scores.append(pred['scores'] * weight)
                if pred['masks'] is not None and len(pred['masks']) > 0:
                    all_masks.append(pred['masks'])
        
        if not all_boxes:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'labels': np.array([]),
                'scores': np.array([]),
                'masks': np.array([]).reshape(0, 1, 1, 1)
            }
        
        # Concatenate and average
        ensembled_boxes = np.concatenate(all_boxes, axis=0)
        ensembled_labels = np.concatenate(all_labels, axis=0)
        ensembled_scores = np.concatenate(all_scores, axis=0)
        
        ensembled_masks = None
        if all_masks:
            ensembled_masks = np.concatenate(all_masks, axis=0)
        
        return {
            'boxes': ensembled_boxes,
            'labels': ensembled_labels,
            'scores': ensembled_scores,
            'masks': ensembled_masks
        }


# Import torchvision here to avoid circular imports
import torchvision
import torchvision.ops
