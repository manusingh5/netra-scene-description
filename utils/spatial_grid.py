# utils/spatial_grid.py
"""
NETRA 3×3 Spatial Grid Mapper
Maps object positions to 9-region grid
"""

from typing import List, Dict

class SpatialGrid:
    """
    Maps object bounding boxes to a 3×3 spatial grid:
    [Top-Left] [Top-Center] [Top-Right]
    [Mid-Left] [Center]     [Mid-Right]
    [Bot-Left] [Bot-Center] [Bot-Right]
    """

    def __init__(self):
        self.grid_size = 3  # 3×3

    def map_objects(self, detections: List[Dict], frame_shape: tuple) -> Dict[str, List[str]]:
        """
        Map detected objects to spatial grid positions

        Args:
            detections: List of YOLO detections (with bbox: [x1, y1, x2, y2])
            frame_shape: (height, width, channels) of the frame

        Returns:
            Dictionary mapping grid regions to lists of object names:
            {
                "top-left": ["person", "car"],
                "center": ["desk"],
                ...
            }
        """
        if not detections or len(frame_shape) < 2:
            return {}

        height, width = frame_shape[:2]
        
        # Initialize grid
        grid = {
            "top-left": [],
            "top-center": [],
            "top-right": [],
            "mid-left": [],
            "center": [],
            "mid-right": [],
            "bot-left": [],
            "bot-center": [],
            "bot-right": []
        }

        # Cell dimensions
        cell_w = width // 3
        cell_h = height // 3

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Determine grid region
            col = min(int(center_x // cell_w), 2)  # 0, 1, or 2
            row = min(int(center_y // cell_h), 2)  # 0, 1, or 2

            # Map to region name
            regions = [
                ["top-left", "top-center", "top-right"],
                ["mid-left", "center", "mid-right"],
                ["bot-left", "bot-center", "bot-right"]
            ]
            
            region = regions[row][col]
            grid[region].append(det.get("name", "unknown"))

        return grid

    def grid_to_text(self, grid: Dict[str, List[str]]) -> str:
        """Convert grid to human-readable text"""
        if not grid:
            return "No spatial information."

        lines = []
        for region, objects in grid.items():
            if objects:
                lines.append(f"{region.replace('-', ' ').title()}: {', '.join(objects)}")
        
        return "; ".join(lines) if lines else "Objects centered in view."