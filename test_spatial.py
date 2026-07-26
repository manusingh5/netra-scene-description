# test_spatial.py
from utils.spatial_grid import SpatialGrid

grid_mapper = SpatialGrid()

# Mock detections
detections = [
    {"name": "person", "bbox": [100, 100, 200, 300]},
    {"name": "desk", "bbox": [400, 400, 600, 600]},
    {"name": "laptop", "bbox": [500, 450, 550, 500]}
]

frame_shape = (800, 1200, 3)  # height, width, channels

result = grid_mapper.map_objects(detections, frame_shape)
print("Spatial Grid:", result)

text = grid_mapper.grid_to_text(result)
print("Text:", text)

print("✅ Spatial grid working!")