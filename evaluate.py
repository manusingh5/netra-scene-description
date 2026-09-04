# evaluate.py
"""
NETRA Evaluation Script
Calculates REAL metrics by comparing pipeline output vs Ground Truth

Run: python evaluate.py
"""

import cv2
import os
import sys
import types
import importlib
import importlib.machinery
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ground_truth import GROUND_TRUTH

# ═══════════════════════════════════════════════════════════════
# FLASH ATTENTION STUB (Fix for ImportError)
# ═══════════════════════════════════════════════════════════════
if importlib.util.find_spec("flash_attn") is None:
    flash_attn_stub = types.ModuleType("flash_attn")
    flash_attn_stub.__version__ = "2.0"
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None, is_package=True
    )
    bert_pad_stub = types.ModuleType("flash_attn.bert_pad")
    bert_pad_stub.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn.bert_pad", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub
    sys.modules["flash_attn.bert_pad"] = bert_pad_stub
    print("  ⚙️ flash_attn stub created (with __spec__)")

# Patch transformers flash_attn check
import transformers.utils.import_utils
_original_is_package_available = transformers.utils.import_utils._is_package_available
def _patched_is_package_available(pkg_name):
    if pkg_name == "flash_attn":
        return True
    return _original_is_package_available(pkg_name)
transformers.utils.import_utils._is_package_available = _patched_is_package_available
transformers.utils.import_utils.is_flash_attn_2_available = lambda: False

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
UPLOADS_DIR = "uploads"
CONFIDENCE_THRESHOLD = 0.5
REPORT_FILE = "evaluation_report.txt"

# ═══════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════

def calc_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def caption_similarity(generated, expected):
    """Simple word overlap ratio between generated and expected caption"""
    gen_words = set(generated.lower().replace('.', '').replace(',', '').split())
    exp_words = set(expected.lower().replace('.', '').replace(',', '').split())
    if len(exp_words) == 0:
        return 0.0
    overlap = gen_words & exp_words
    return len(overlap) / len(exp_words)

def text_accuracy(detected_texts, expected_texts):
    """Word-level accuracy for OCR"""
    if len(expected_texts) == 0:
        return 1.0 if len(detected_texts) == 0 else 0.0
    correct = 0
    for exp in expected_texts:
        for det in detected_texts:
            if exp.lower() in det.lower() or det.lower() in exp.lower():
                correct += 1
                break
    return correct / len(expected_texts)

# ═══════════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_frame_at_timestamp(video_path, timestamp_sec):
    """Extract a single frame at given timestamp"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("NETRA EVALUATION — Starting...")
    print("=" * 60)

    # ── Load Models ──
    print("\n[1/4] Loading models...")

    # YOLO
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
    print("  ✅ YOLOv8-nano loaded")

    # Florence-2 — Direct loading with flash_attn stub
    import torch
    import PIL.Image as Image
    from transformers import AutoProcessor, AutoModelForCausalLM

    florence_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True
    )
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    florence_model.eval()
    print("  ✅ Florence-2 loaded")

    # PaddleOCR
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("  ✅ PaddleOCR loaded")

    print("\n[2/4] All models loaded. Starting evaluation...\n")

    # ── Evaluation Variables ──
    yolo_tp, yolo_fp, yolo_fn = 0, 0, 0
    per_class_stats = {}
    caption_scores = []
    ocr_scores = []
    total_frames = 0

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("NETRA EVALUATION REPORT")
    report_lines.append("=" * 60)

    # ── Process Each Video ──
    for video_name, video_data in GROUND_TRUTH.items():
        video_path = os.path.join(UPLOADS_DIR, video_name)

        if not os.path.exists(video_path):
            print(f"  ⚠️ Video not found: {video_name}")
            continue

        print(f"\n{'─' * 50}")
        print(f"Video: {video_name}")
        print(f"   Scene: {video_data['scene_type']}")
        report_lines.append(f"\n{'─' * 50}")
        report_lines.append(f"Video: {video_name}")
        report_lines.append(f"Scene: {video_data['scene_type']}")

        frames = video_data["frames"]

        for timestamp, gt_data in frames.items():
            total_frames += 1
            print(f"\n  Timestamp: {timestamp}s")

            # Extract frame
            frame = extract_frame_at_timestamp(video_path, timestamp)
            if frame is None:
                print(f"     ⚠️ Could not extract frame at {timestamp}s")
                continue

            # ─── YOLO EVALUATION ───
            yolo_results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            detected_objects = []

            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]
                    conf = float(box.conf[0])
                    detected_objects.append({"class": cls_name, "confidence": conf})

            # Compare with GT
            gt_objects = gt_data["objects"]
            det_counts = Counter([d["class"] for d in detected_objects])
            gt_counts = Counter(gt_objects)

            all_classes_in_frame = set(list(det_counts.keys()) + list(gt_counts.keys()))

            frame_tp, frame_fp, frame_fn = 0, 0, 0

            for cls in all_classes_in_frame:
                det = det_counts.get(cls, 0)
                gt = gt_counts.get(cls, 0)

                tp = min(det, gt)
                fp = max(0, det - gt)
                fn = max(0, gt - det)

                frame_tp += tp
                frame_fp += fp
                frame_fn += fn

                if cls not in per_class_stats:
                    per_class_stats[cls] = {"tp": 0, "fp": 0, "fn": 0}
                per_class_stats[cls]["tp"] += tp
                per_class_stats[cls]["fp"] += fp
                per_class_stats[cls]["fn"] += fn

            yolo_tp += frame_tp
            yolo_fp += frame_fp
            yolo_fn += frame_fn

            yolo_prec, yolo_rec, yolo_f1 = calc_precision_recall_f1(frame_tp, frame_fp, frame_fn)

            print(f"     YOLO -> GT: {gt_objects}")
            print(f"     YOLO -> Detected: {[d['class'] for d in detected_objects]}")
            print(f"     YOLO -> Frame P/R/F1: {yolo_prec:.2f}/{yolo_rec:.2f}/{yolo_f1:.2f}")

            # ─── FLORENCE-2 EVALUATION ───
            pil_image = Image.fromarray(frame)

            inputs = florence_processor(
                text="<CAPTION>", images=pil_image, return_tensors="pt"
            )
            with torch.no_grad():
                generated = florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            generated_text = florence_processor.batch_decode(generated, skip_special_tokens=True)[0]

            gt_caption = gt_data["expected_caption"]
            cap_score = caption_similarity(generated_text, gt_caption)
            caption_scores.append(cap_score)

            print(f"     Florence-2 -> Generated: {generated_text}")
            print(f"     Florence-2 -> Expected:  {gt_caption}")
            print(f"     Florence-2 -> Similarity: {cap_score:.2f}")

            # ─── PADDLEOCR EVALUATION ───
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            try:
                ocr_result = ocr_engine.ocr(frame_bgr, cls=True)
            except Exception as e:
                print(f"     OCR error: {e}")
                ocr_result = None

            detected_texts = []
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        detected_texts.append(text)

            gt_texts = gt_data["expected_text"]
            ocr_acc = text_accuracy(detected_texts, gt_texts)
            ocr_scores.append(ocr_acc)

            print(f"     PaddleOCR -> Detected: {detected_texts}")
            print(f"     PaddleOCR -> Expected: {gt_texts}")
            print(f"     PaddleOCR -> Accuracy: {ocr_acc:.2f}")

            # ─── Write to report ───
            report_lines.append(f"\n  Frame @ {timestamp}s:")
            report_lines.append(f"    YOLO:")
            report_lines.append(f"      Ground Truth:   {gt_objects}")
            report_lines.append(f"      Detected:       {[d['class'] for d in detected_objects]}")
            report_lines.append(f"      Frame P/R/F1:   {yolo_prec:.2f} / {yolo_rec:.2f} / {yolo_f1:.2f}")
            report_lines.append(f"    Florence-2:")
            report_lines.append(f"      Generated:  {generated_text}")
            report_lines.append(f"      Expected:   {gt_caption}")
            report_lines.append(f"      Similarity: {cap_score:.2f}")
            report_lines.append(f"    PaddleOCR:")
            report_lines.append(f"      Detected:  {detected_texts}")
            report_lines.append(f"      Expected:  {gt_texts}")
            report_lines.append(f"      Accuracy:  {ocr_acc:.2f}")

    # ── FINAL METRICS CALCULATION ──
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")

    # YOLO Overall
    yolo_precision, yolo_recall, yolo_f1 = calc_precision_recall_f1(yolo_tp, yolo_fp, yolo_fn)
    print(f"\nYOLOv8 Detection (Overall):")
    print(f"   True Positives:  {yolo_tp}")
    print(f"   False Positives: {yolo_fp}")
    print(f"   False Negatives: {yolo_fn}")
    print(f"   Precision: {yolo_precision*100:.1f}%")
    print(f"   Recall:    {yolo_recall*100:.1f}%")
    print(f"   F1 Score:  {yolo_f1*100:.1f}%")

    # Per-class YOLO
    print(f"\nPer-Class Breakdown:")
    for cls, stats in sorted(per_class_stats.items()):
        p, r, f1 = calc_precision_recall_f1(stats["tp"], stats["fp"], stats["fn"])
        print(f"   {cls:20s}: P={p*100:.1f}% R={r*100:.1f}% F1={f1*100:.1f}% (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})")

    # Florence-2
    avg_caption_score = sum(caption_scores) / len(caption_scores) if caption_scores else 0
    print(f"\nFlorence-2 Captioning:")
    print(f"   Avg Similarity Score: {avg_caption_score*100:.1f}%")
    print(f"   Frames Evaluated:     {len(caption_scores)}")
    print(f"   Scores: {[f'{s:.2f}' for s in caption_scores]}")

    # PaddleOCR
    avg_ocr_score = sum(ocr_scores) / len(ocr_scores) if ocr_scores else 0
    print(f"\nPaddleOCR:")
    print(f"   Avg Word Accuracy: {avg_ocr_score*100:.1f}%")
    print(f"   Frames Evaluated:  {len(ocr_scores)}")
    print(f"   Scores: {[f'{s:.2f}' for s in ocr_scores]}")

    # Overall
    overall = (yolo_f1 + avg_caption_score + avg_ocr_score) / 3
    print(f"\n{'=' * 60}")
    print(f"OVERALL PIPELINE SCORE: {overall*100:.1f}%")
    print(f"{'=' * 60}")
    print(f"   YOLO F1:           {yolo_f1*100:.1f}%")
    print(f"   Florence-2 Score:  {avg_caption_score*100:.1f}%")
    print(f"   PaddleOCR Score:   {avg_ocr_score*100:.1f}%")
    print(f"   Total Frames:      {total_frames}")
    print(f"{'=' * 60}")

    # ── Write report file ──
    report_lines.append(f"\n{'=' * 60}")
    report_lines.append("FINAL SUMMARY")
    report_lines.append(f"{'=' * 60}")
    report_lines.append(f"")
    report_lines.append(f"YOLOv8 Detection:")
    report_lines.append(f"  TP={yolo_tp}, FP={yolo_fp}, FN={yolo_fn}")
    report_lines.append(f"  Precision: {yolo_precision*100:.1f}%")
    report_lines.append(f"  Recall:    {yolo_recall*100:.1f}%")
    report_lines.append(f"  F1 Score:  {yolo_f1*100:.1f}%")
    report_lines.append(f"")
    report_lines.append(f"Per-Class Breakdown:")
    for cls, stats in sorted(per_class_stats.items()):
        p, r, f1 = calc_precision_recall_f1(stats["tp"], stats["fp"], stats["fn"])
        report_lines.append(f"  {cls}: P={p*100:.1f}% R={r*100:.1f}% F1={f1*100:.1f}%")
    report_lines.append(f"")
    report_lines.append(f"Florence-2 Captioning:")
    report_lines.append(f"  Avg Similarity: {avg_caption_score*100:.1f}%")
    report_lines.append(f"")
    report_lines.append(f"PaddleOCR:")
    report_lines.append(f"  Avg Word Accuracy: {avg_ocr_score*100:.1f}%")
    report_lines.append(f"")
    report_lines.append(f"OVERALL PIPELINE SCORE: {overall*100:.1f}%")
    report_lines.append(f"Total Frames Evaluated: {total_frames}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"\nEvaluation Complete!")

if __name__ == "__main__":
    main()