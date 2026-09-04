# ground_truth.py
# NETRA Evaluation — Ground Truth Annotations

GROUND_TRUTH = {
    "test_video1.mp4": {
        "duration_sec": 45,
        "scene_type": "City street / downtown road",
        "frames": {
            0: {
                "objects": ["person", "car", "car", "bus", "building", "tree", "traffic_light"],
                "expected_caption": "A busy city street with several cars, a bus, pedestrians, and tall buildings.",
                "expected_text": ["E Washington St", "ONE WAY"]
            },
            10: {
                "objects": ["person", "car", "taxi", "bus", "building", "tree"],
                "expected_caption": "A busy downtown street with a bus, several cars, a taxi, and pedestrians.",
                "expected_text": ["E Washington St", "ONE WAY", "151 TO UNION STATION"]
            },
            20: {
                "objects": ["bus", "car", "person", "building", "tree", "traffic_light"],
                "expected_caption": "A city bus is stopped on a busy downtown street with cars and pedestrians around it.",
                "expected_text": ["E Washington St", "ONE WAY", "151 TO UNION STATION"]
            },
            30: {
                "objects": ["bus", "car", "person", "building", "traffic_light"],
                "expected_caption": "A city bus is at a bus stop on a downtown street with several cars and pedestrians.",
                "expected_text": ["E Washington St", "ONE WAY", "151 TO UNION STATION"]
            },
            40: {
                "objects": ["bus", "person", "car", "building", "tree"],
                "expected_caption": "A large city bus passes close to the camera on a busy downtown street.",
                "expected_text": ["E Washington St", "ONE WAY", "transitchicago.com"]
            }
        }
    },
    "test_video2.mp4": {
        "duration_sec": 13,
        "scene_type": "Airport terminal",
        "frames": {
            0: {
                "objects": ["person", "person", "person", "suitcase", "handbag"],
                "expected_caption": "People walk through an airport terminal with luggage and luggage carts.",
                "expected_text": ["Emergency Exit", "Gates", "41-46", "51-58", "71-78"]
            },
            10: {
                "objects": ["person", "person", "suitcase", "handbag"],
                "expected_caption": "People are walking through a busy airport terminal toward the gates.",
                "expected_text": ["Emergency Exit", "Gates", "41-46", "51-58", "71-78"]
            }
        }
    }
}


def get_all_videos():
    """Return list of video filenames in ground truth"""
    return list(GROUND_TRUTH.keys())


def get_frames(video_name):
    """Return dict of {timestamp: frame_data} for a video"""
    if video_name in GROUND_TRUTH:
        return GROUND_TRUTH[video_name]["frames"]
    return {}


def get_all_classes():
    """Return all unique object class names across all videos"""
    classes = set()
    for video_data in GROUND_TRUTH.values():
        for frame_data in video_data["frames"].values():
            for obj in frame_data["objects"]:
                classes.add(obj)
    return sorted(list(classes))


def get_all_text():
    """Return all unique text strings across all videos"""
    texts = set()
    for video_data in GROUND_TRUTH.values():
        for frame_data in video_data["frames"].values():
            for txt in frame_data["expected_text"]:
                texts.add(txt)
    return sorted(list(texts))