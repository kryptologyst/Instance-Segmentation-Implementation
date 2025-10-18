"""
Synthetic data generation for instance segmentation testing.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic images with known objects for testing instance segmentation."""
    
    def __init__(self, image_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Initialize the synthetic data generator.
        
        Args:
            image_size: Size of generated images (width, height)
        """
        self.image_size = image_size
        self.objects = self._define_objects()
    
    def _define_objects(self) -> List[dict]:
        """Define synthetic objects to draw."""
        return [
            {"name": "person", "color": (255, 0, 0), "size_range": (50, 100)},
            {"name": "car", "color": (0, 255, 0), "size_range": (80, 150)},
            {"name": "bicycle", "color": (0, 0, 255), "size_range": (40, 80)},
            {"name": "dog", "color": (255, 255, 0), "size_range": (30, 60)},
            {"name": "chair", "color": (255, 0, 255), "size_range": (40, 80)},
            {"name": "bottle", "color": (0, 255, 255), "size_range": (20, 40)},
            {"name": "laptop", "color": (128, 128, 128), "size_range": (60, 100)},
            {"name": "book", "color": (255, 128, 0), "size_range": (30, 50)},
        ]
    
    def generate_image(self, num_objects: int = 5, background_color: Tuple[int, int, int] = (240, 240, 240)) -> Tuple[np.ndarray, List[dict]]:
        """
        Generate a synthetic image with random objects.
        
        Args:
            num_objects: Number of objects to place in the image
            background_color: Background color (RGB)
            
        Returns:
            Tuple of (image_array, object_annotations)
        """
        # Create blank image
        image = Image.new('RGB', self.image_size, background_color)
        draw = ImageDraw.Draw(image)
        
        annotations = []
        placed_objects = []
        
        for i in range(num_objects):
            # Select random object type
            obj_type = random.choice(self.objects)
            
            # Generate random position and size
            size = random.randint(*obj_type["size_range"])
            x = random.randint(0, self.image_size[0] - size)
            y = random.randint(0, self.image_size[1] - size)
            
            # Check for overlaps (simple collision detection)
            new_rect = (x, y, x + size, y + size)
            overlap = False
            for placed_rect in placed_objects:
                if self._rectangles_overlap(new_rect, placed_rect):
                    overlap = True
                    break
            
            if overlap:
                continue  # Skip this object if it overlaps
            
            placed_objects.append(new_rect)
            
            # Draw object based on type
            if obj_type["name"] == "person":
                self._draw_person(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "car":
                self._draw_car(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "bicycle":
                self._draw_bicycle(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "dog":
                self._draw_dog(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "chair":
                self._draw_chair(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "bottle":
                self._draw_bottle(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "laptop":
                self._draw_laptop(draw, x, y, size, obj_type["color"])
            elif obj_type["name"] == "book":
                self._draw_book(draw, x, y, size, obj_type["color"])
            
            # Add annotation
            annotations.append({
                "bbox": [x, y, x + size, y + size],
                "class": obj_type["name"],
                "confidence": random.uniform(0.8, 1.0)
            })
        
        return np.array(image), annotations
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)
    
    def _draw_person(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple person shape."""
        # Head
        head_size = size // 6
        draw.ellipse([x + size//2 - head_size//2, y, x + size//2 + head_size//2, y + head_size], fill=color)
        
        # Body
        body_width = size // 4
        body_height = size // 2
        draw.rectangle([x + size//2 - body_width//2, y + head_size, x + size//2 + body_width//2, y + head_size + body_height], fill=color)
        
        # Arms
        arm_width = size // 8
        arm_height = size // 3
        draw.rectangle([x, y + head_size + size//8, x + arm_width, y + head_size + size//8 + arm_height], fill=color)
        draw.rectangle([x + size - arm_width, y + head_size + size//8, x + size, y + head_size + size//8 + arm_height], fill=color)
        
        # Legs
        leg_width = size // 6
        leg_height = size // 3
        draw.rectangle([x + size//2 - leg_width, y + head_size + body_height, x + size//2, y + head_size + body_height + leg_height], fill=color)
        draw.rectangle([x + size//2, y + head_size + body_height, x + size//2 + leg_width, y + head_size + body_height + leg_height], fill=color)
    
    def _draw_car(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple car shape."""
        # Car body
        draw.rectangle([x, y + size//3, x + size, y + size], fill=color)
        
        # Windows
        window_color = (200, 200, 200)
        draw.rectangle([x + size//8, y + size//4, x + size - size//8, y + size//2], fill=window_color)
        
        # Wheels
        wheel_color = (50, 50, 50)
        wheel_size = size // 6
        draw.ellipse([x + size//6 - wheel_size//2, y + size - wheel_size//2, x + size//6 + wheel_size//2, y + size + wheel_size//2], fill=wheel_color)
        draw.ellipse([x + size - size//6 - wheel_size//2, y + size - wheel_size//2, x + size - size//6 + wheel_size//2, y + size + wheel_size//2], fill=wheel_color)
    
    def _draw_bicycle(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple bicycle shape."""
        # Wheels
        wheel_size = size // 3
        draw.ellipse([x, y + size//2 - wheel_size//2, x + wheel_size, y + size//2 + wheel_size//2], fill=color)
        draw.ellipse([x + size - wheel_size, y + size//2 - wheel_size//2, x + size, y + size//2 + wheel_size//2], fill=color)
        
        # Frame
        frame_width = 3
        draw.line([x + wheel_size//2, y + size//2, x + size - wheel_size//2, y + size//2], fill=color, width=frame_width)
        draw.line([x + wheel_size//2, y + size//2, x + size//2, y + size//4], fill=color, width=frame_width)
        draw.line([x + size//2, y + size//4, x + size - wheel_size//2, y + size//2], fill=color, width=frame_width)
    
    def _draw_dog(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple dog shape."""
        # Body
        body_width = size // 2
        body_height = size // 2
        draw.ellipse([x + size//2 - body_width//2, y + size//3, x + size//2 + body_width//2, y + size//3 + body_height], fill=color)
        
        # Head
        head_size = size // 3
        draw.ellipse([x + size//2 - head_size//2, y, x + size//2 + head_size//2, y + head_size], fill=color)
        
        # Ears
        ear_size = size // 8
        draw.ellipse([x + size//2 - head_size//2, y, x + size//2 - head_size//2 + ear_size, y + ear_size], fill=color)
        draw.ellipse([x + size//2 + head_size//2 - ear_size, y, x + size//2 + head_size//2, y + ear_size], fill=color)
        
        # Tail
        tail_width = size // 8
        tail_height = size // 3
        draw.rectangle([x + size - tail_width, y + size//3, x + size, y + size//3 + tail_height], fill=color)
    
    def _draw_chair(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple chair shape."""
        # Seat
        seat_height = size // 4
        draw.rectangle([x + size//8, y + size//3, x + size - size//8, y + size//3 + seat_height], fill=color)
        
        # Back
        back_width = size // 8
        back_height = size // 2
        draw.rectangle([x + size - size//4, y, x + size - size//4 + back_width, y + size//3 + seat_height], fill=color)
        
        # Legs
        leg_width = size // 16
        leg_height = size // 3
        draw.rectangle([x + size//8, y + size//3 + seat_height, x + size//8 + leg_width, y + size//3 + seat_height + leg_height], fill=color)
        draw.rectangle([x + size - size//8 - leg_width, y + size//3 + seat_height, x + size - size//8, y + size//3 + seat_height + leg_height], fill=color)
    
    def _draw_bottle(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple bottle shape."""
        # Bottle body
        body_width = size // 3
        body_height = size // 2
        draw.rectangle([x + size//2 - body_width//2, y + size//4, x + size//2 + body_width//2, y + size//4 + body_height], fill=color)
        
        # Bottle neck
        neck_width = size // 6
        neck_height = size // 4
        draw.rectangle([x + size//2 - neck_width//2, y, x + size//2 + neck_width//2, y + neck_height], fill=color)
        
        # Cap
        cap_size = size // 5
        draw.rectangle([x + size//2 - cap_size//2, y - size//8, x + size//2 + cap_size//2, y], fill=(100, 100, 100))
    
    def _draw_laptop(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple laptop shape."""
        # Screen
        screen_width = size // 2
        screen_height = size // 2
        draw.rectangle([x + size//4, y, x + size//4 + screen_width, y + screen_height], fill=color)
        
        # Keyboard
        keyboard_width = size
        keyboard_height = size // 4
        draw.rectangle([x, y + screen_height, x + keyboard_width, y + screen_height + keyboard_height], fill=(200, 200, 200))
        
        # Screen content
        screen_color = (50, 50, 50)
        draw.rectangle([x + size//4 + size//16, y + size//16, x + size//4 + screen_width - size//16, y + screen_height - size//16], fill=screen_color)
    
    def _draw_book(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple book shape."""
        # Book cover
        draw.rectangle([x, y, x + size, y + size], fill=color)
        
        # Pages
        page_color = (250, 250, 250)
        draw.rectangle([x + size//8, y + size//8, x + size - size//8, y + size - size//8], fill=page_color)
        
        # Text lines
        text_color = (100, 100, 100)
        line_height = size // 8
        for i in range(3):
            line_y = y + size//4 + i * line_height
            draw.line([x + size//4, line_y, x + size - size//4, line_y], fill=text_color, width=1)
    
    def generate_dataset(self, num_images: int, output_dir: Union[str, Path]) -> None:
        """
        Generate a dataset of synthetic images.
        
        Args:
            num_images: Number of images to generate
            output_dir: Directory to save images and annotations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating {num_images} synthetic images...")
        
        for i in range(num_images):
            # Generate random number of objects
            num_objects = random.randint(2, 8)
            
            # Generate image and annotations
            image, annotations = self.generate_image(num_objects)
            
            # Save image
            image_path = images_dir / f"synthetic_{i:03d}.jpg"
            Image.fromarray(image).save(image_path)
            
            # Save annotations
            annotation_path = annotations_dir / f"synthetic_{i:03d}.json"
            import json
            with open(annotation_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_images} images")
        
        logger.info(f"Dataset generation complete. Saved to {output_dir}")


def main() -> None:
    """Generate sample synthetic images for testing."""
    generator = SyntheticDataGenerator()
    
    # Create sample images
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a few sample images
    for i in range(5):
        image, annotations = generator.generate_image(num_objects=random.randint(3, 6))
        
        # Save image
        image_path = sample_dir / f"sample_{i:02d}.jpg"
        Image.fromarray(image).save(image_path)
        
        # Save annotations
        annotation_path = sample_dir / f"sample_{i:02d}.json"
        import json
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Generated sample {i+1}: {image_path}")
    
    print(f"Sample images generated in {sample_dir}")


if __name__ == "__main__":
    main()
