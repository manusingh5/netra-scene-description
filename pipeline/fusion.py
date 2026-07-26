# pipeline/fusion.py
"""
NETRA Descriptor Fusion
Merges per-frame descriptors and deduplicates
"""

from typing import List, Dict

class DescriptorFusion:
    """
    Fuses all frame descriptors into ONE consolidated scene descriptor
    """

    def __init__(self, similarity_threshold: float = 0.50):
        self.similarity_threshold = similarity_threshold

    def fuse(self, descriptors: List[Dict]) -> List[Dict]:
        """Merge all descriptors into ONE unified scene"""
        if not descriptors:
            return []

        print(f"[FUSION] Input: {len(descriptors)} frame descriptors")

        # Instead of grouping, merge EVERYTHING into one scene
        merged = self._merge_all(descriptors)

        print(f"[FUSION] Output: 1 unified scene descriptor")
        return [merged]

    def _merge_all(self, descriptors: List[Dict]) -> Dict:
        """Merge all frame descriptors into one"""
        all_objects = {}
        all_texts = set()
        all_captions = []

        for desc in descriptors:
            # Collect objects (keep highest confidence)
            for obj in desc.get("objects", []):
                key = obj["name"]
                if key not in all_objects or obj.get("confidence", 0) > all_objects[key].get("confidence", 0):
                    all_objects[key] = obj

            # Collect unique texts
            for text in desc.get("text_detected", []):
                all_texts.add(text)

            # Collect non-empty captions
            caption = desc.get("caption", "")
            if caption and caption not in ["[Description unavailable]", "[Skipped for speed]"]:
                all_captions.append(caption)

        # Best caption (first valid one)
        best_caption = all_captions[0] if all_captions else ""

        return {
            "caption": best_caption,
            "objects": list(all_objects.values()),
            "text_detected": list(all_texts),
            "spatial_map": descriptors[-1].get("spatial_map", {}) if descriptors else {},
            "time_range": [descriptors[0].get("timestamp", 0), descriptors[-1].get("timestamp", 0)] if descriptors else [0, 0],
            "frame_ids": [d.get("frame_id", 0) for d in descriptors],
            "frame_count": len(descriptors)
        }