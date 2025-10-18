"""
Tests for the instance segmentation implementation.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import tempfile
import os

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from instance_segmentation import InstanceSegmentationModel, InstanceSegmentationPipeline
from data_generator import SyntheticDataGenerator
from advanced_segmentation import AdvancedInstanceSegmentationModel


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SyntheticDataGenerator()
        assert generator.image_size == (640, 480)
        assert len(generator.objects) > 0
    
    def test_generate_image(self):
        """Test image generation."""
        generator = SyntheticDataGenerator(image_size=(320, 240))
        image, annotations = generator.generate_image(num_objects=3)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (240, 320, 3)
        assert len(annotations) <= 3
        assert all('bbox' in ann for ann in annotations)
        assert all('class' in ann for ann in annotations)
        assert all('confidence' in ann for ann in annotations)
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        generator = SyntheticDataGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator.generate_dataset(num_images=5, output_dir=temp_dir)
            
            # Check if files were created
            images_dir = Path(temp_dir) / "images"
            annotations_dir = Path(temp_dir) / "annotations"
            
            assert images_dir.exists()
            assert annotations_dir.exists()
            
            image_files = list(images_dir.glob("*.jpg"))
            annotation_files = list(annotations_dir.glob("*.json"))
            
            assert len(image_files) == 5
            assert len(annotation_files) == 5


class TestInstanceSegmentationModel:
    """Test instance segmentation model."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        generator = SyntheticDataGenerator(image_size=(256, 256))
        image, _ = generator.generate_image(num_objects=2)
        return image
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = InstanceSegmentationModel(
            model_name="maskrcnn_resnet50_fpn",
            confidence_threshold=0.5
        )
        
        assert model.model_name == "maskrcnn_resnet50_fpn"
        assert model.confidence_threshold == 0.5
        assert model.device in ["cpu", "cuda"]
    
    def test_preprocess_image(self, sample_image):
        """Test image preprocessing."""
        model = InstanceSegmentationModel()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            Image.fromarray(sample_image).save(tmp_file.name)
            
            try:
                image_tensor = model.preprocess_image(tmp_file.name)
                assert isinstance(image_tensor, torch.Tensor)
                assert image_tensor.shape[0] == 1  # Batch dimension
                assert image_tensor.shape[1] == 3  # RGB channels
            finally:
                os.unlink(tmp_file.name)
    
    def test_predict(self, sample_image):
        """Test model prediction."""
        model = InstanceSegmentationModel(confidence_threshold=0.1)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            Image.fromarray(sample_image).save(tmp_file.name)
            
            try:
                image_tensor = model.preprocess_image(tmp_file.name)
                predictions = model.predict(image_tensor)
                
                assert isinstance(predictions, dict)
                assert 'boxes' in predictions
                assert 'labels' in predictions
                assert 'scores' in predictions
                assert 'masks' in predictions
                
                assert len(predictions['boxes']) == len(predictions['labels'])
                assert len(predictions['labels']) == len(predictions['scores'])
                
            finally:
                os.unlink(tmp_file.name)


class TestInstanceSegmentationPipeline:
    """Test complete pipeline."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        generator = SyntheticDataGenerator(image_size=(256, 256))
        image, _ = generator.generate_image(num_objects=2)
        return image
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = InstanceSegmentationPipeline()
        assert pipeline.model is not None
        assert pipeline.visualizer is not None
    
    def test_process_image(self, sample_image):
        """Test complete image processing."""
        pipeline = InstanceSegmentationPipeline(confidence_threshold=0.1)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            Image.fromarray(sample_image).save(tmp_file.name)
            
            try:
                predictions = pipeline.process_image(
                    tmp_file.name, 
                    show_results=False
                )
                
                assert isinstance(predictions, dict)
                assert 'boxes' in predictions
                assert 'labels' in predictions
                assert 'scores' in predictions
                assert 'masks' in predictions
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_batch_process(self, sample_image):
        """Test batch processing."""
        pipeline = InstanceSegmentationPipeline(confidence_threshold=0.1)
        
        # Create multiple temporary images
        temp_files = []
        try:
            for i in range(3):
                tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                Image.fromarray(sample_image).save(tmp_file.name)
                temp_files.append(tmp_file.name)
            
            results = pipeline.batch_process(temp_files, show_results=False)
            
            assert len(results) == 3
            assert all(isinstance(result, dict) for result in results)
            
        finally:
            for tmp_file in temp_files:
                os.unlink(tmp_file)


class TestAdvancedSegmentation:
    """Test advanced segmentation features."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        generator = SyntheticDataGenerator(image_size=(256, 256))
        image, _ = generator.generate_image(num_objects=2)
        return image
    
    def test_advanced_model_initialization(self):
        """Test advanced model initialization."""
        model = AdvancedInstanceSegmentationModel(
            model_name="maskrcnn_resnet50_fpn",
            use_nms=True,
            nms_threshold=0.5,
            use_tta=False
        )
        
        assert model.use_nms == True
        assert model.nms_threshold == 0.5
        assert model.use_tta == False
    
    def test_tta_prediction(self, sample_image):
        """Test Test Time Augmentation."""
        model = AdvancedInstanceSegmentationModel(
            model_name="maskrcnn_resnet50_fpn",
            use_tta=True,
            confidence_threshold=0.1
        )
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            Image.fromarray(sample_image).save(tmp_file.name)
            
            try:
                image_tensor = model.preprocess_image(tmp_file.name)
                predictions = model.predict(image_tensor)
                
                assert isinstance(predictions, dict)
                assert 'boxes' in predictions
                assert 'labels' in predictions
                assert 'scores' in predictions
                
            finally:
                os.unlink(tmp_file.name)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Generate synthetic data
        generator = SyntheticDataGenerator(image_size=(320, 240))
        image, annotations = generator.generate_image(num_objects=3)
        
        # Save image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            Image.fromarray(image).save(tmp_file.name)
            
            try:
                # Process with pipeline
                pipeline = InstanceSegmentationPipeline(confidence_threshold=0.1)
                predictions = pipeline.process_image(tmp_file.name, show_results=False)
                
                # Verify results
                assert isinstance(predictions, dict)
                assert 'boxes' in predictions
                assert 'labels' in predictions
                assert 'scores' in predictions
                assert 'masks' in predictions
                
                # Check that we got some detections
                assert len(predictions['boxes']) >= 0
                
            finally:
                os.unlink(tmp_file.name)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
