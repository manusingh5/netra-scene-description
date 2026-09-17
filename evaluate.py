"""
NETRA Evaluation Script

Metrics:
1. YOLOv8
   - Precision
   - Recall
   - F1 Score

   NOTE:
   These metrics use class/count matching.
   Proper bounding-box mAP is evaluated separately using
   Ultralytics validation and manually annotated bounding boxes.

2. Florence-2
   - BLEU
   - ROUGE-L
   - BERTScore Precision
   - BERTScore Recall
   - BERTScore F1

3. PaddleOCR
   - Raw CER
   - Raw WER
   - Normalized Order-Invariant CER
   - Normalized Order-Invariant WER

Run:
    python evaluate.py
"""

import os
import sys
import re
import unicodedata
import cv2

from collections import Counter

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ground_truth import GROUND_TRUTH

from models.yolo_model import YOLOModel
from models.ocr_model import OCRModel
from models.florence_model import FlorenceModel

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

from rouge_score import rouge_scorer
from jiwer import wer, cer
from bert_score import score as bert_score


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

UPLOADS_DIR = os.path.join(
    PROJECT_ROOT,
    "uploads"
)

REPORT_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation_report.txt"
)


# ------------------------------------------------------------
# GENERAL HELPERS
# ------------------------------------------------------------

def calc_precision_recall_f1(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return precision, recall, f1


# ------------------------------------------------------------
# TEXT NORMALIZATION
# ------------------------------------------------------------

def normalize_text(text):
    """
    Basic normalization used for OCR evaluation.

    - Unicode normalization
    - lowercase
    - remove punctuation
    - keep letters/numbers
    - collapse whitespace
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = text.lower()

    # Replace punctuation/symbols with spaces.
    # Keep letters, digits and whitespace.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Underscore is included in \w, remove separately.
    text = text.replace(
        "_",
        " "
    )

    # Collapse repeated spaces.
    text = " ".join(
        text.split()
    )

    return text


def normalize_ocr_regions(texts):
    """
    Normalize every OCR text region independently.

    PaddleOCR may return regions in a different order from
    manually entered ground truth.

    Sorting makes the comparison order-invariant at region level.
    """

    normalized_regions = []

    for text in texts:

        cleaned = normalize_text(
            text
        )

        if cleaned:
            normalized_regions.append(
                cleaned
            )

    normalized_regions.sort()

    return normalized_regions


# ------------------------------------------------------------
# FLORENCE HELPERS
# ------------------------------------------------------------

def clean_caption(text):
    """
    Remove Florence special tokens.
    """

    if not text:
        return ""

    text = text.replace(
        "</s>",
        ""
    )

    text = text.replace(
        "<s>",
        ""
    )

    text = text.strip()

    return text


def calculate_bleu(
    generated_caption,
    reference_caption
):

    if (
        not generated_caption
        or not reference_caption
    ):
        return 0.0

    generated_tokens = (
        generated_caption
        .lower()
        .split()
    )

    reference_tokens = (
        reference_caption
        .lower()
        .split()
    )

    smoothing = (
        SmoothingFunction()
        .method1
    )

    score = sentence_bleu(
        [reference_tokens],
        generated_tokens,
        smoothing_function=smoothing
    )

    return score


def calculate_rouge_l(
    generated_caption,
    reference_caption
):

    if (
        not generated_caption
        or not reference_caption
    ):
        return 0.0

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    result = scorer.score(
        reference_caption,
        generated_caption
    )

    return result[
        "rougeL"
    ].fmeasure


# ------------------------------------------------------------
# OCR METRICS
# ------------------------------------------------------------

def calculate_ocr_metrics(
    detected_texts,
    expected_texts
):
    """
    Returns:

    raw_cer
    raw_wer
    normalized_cer
    normalized_wer

    RAW:
        Directly joins OCR regions.

    NORMALIZED:
        Normalizes each region and sorts regions before joining.

    The normalized version is useful for scene-text OCR because
    OCR detection order is not always the same as manual
    ground-truth order.
    """

    # --------------------------------------------------------
    # RAW TEXT
    # --------------------------------------------------------

    raw_detected = " ".join(
        str(x)
        for x in detected_texts
    )

    raw_expected = " ".join(
        str(x)
        for x in expected_texts
    )

    raw_detected = (
        raw_detected
        .lower()
        .strip()
    )

    raw_expected = (
        raw_expected
        .lower()
        .strip()
    )

    # --------------------------------------------------------
    # RAW CER / WER
    # --------------------------------------------------------

    if not raw_expected:

        if not raw_detected:

            raw_cer = 0.0
            raw_wer = 0.0

        else:

            raw_cer = 1.0
            raw_wer = 1.0

    else:

        try:

            raw_cer = cer(
                raw_expected,
                raw_detected
            )

        except Exception:

            raw_cer = 1.0

        try:

            raw_wer = wer(
                raw_expected,
                raw_detected
            )

        except Exception:

            raw_wer = 1.0


    # --------------------------------------------------------
    # NORMALIZED REGIONS
    # --------------------------------------------------------

    detected_regions = (
        normalize_ocr_regions(
            detected_texts
        )
    )

    expected_regions = (
        normalize_ocr_regions(
            expected_texts
        )
    )

    normalized_detected = " ".join(
        detected_regions
    )

    normalized_expected = " ".join(
        expected_regions
    )

    # --------------------------------------------------------
    # NORMALIZED CER / WER
    # --------------------------------------------------------

    if not normalized_expected:

        if not normalized_detected:

            normalized_cer = 0.0
            normalized_wer = 0.0

        else:

            normalized_cer = 1.0
            normalized_wer = 1.0

    else:

        try:

            normalized_cer = cer(
                normalized_expected,
                normalized_detected
            )

        except Exception:

            normalized_cer = 1.0

        try:

            normalized_wer = wer(
                normalized_expected,
                normalized_detected
            )

        except Exception:

            normalized_wer = 1.0


    return {
        "raw_cer": raw_cer,
        "raw_wer": raw_wer,
        "normalized_cer": normalized_cer,
        "normalized_wer": normalized_wer,
        "raw_expected": raw_expected,
        "raw_detected": raw_detected,
        "normalized_expected": normalized_expected,
        "normalized_detected": normalized_detected
    }


# ------------------------------------------------------------
# FRAME EXTRACTION
# ------------------------------------------------------------

def extract_frame_at_timestamp(
    video_path,
    timestamp_sec
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        cap.release()
        return None

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp_sec * 1000
    )

    success, frame = cap.read()

    cap.release()

    if success:
        return frame

    return None


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print(
        "=" * 70
    )

    print(
        "NETRA MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\n[1/3] Loading NETRA models...\n"
    )


    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    yolo_model = YOLOModel()

    ocr_model = OCRModel()

    florence_model = FlorenceModel()

    print(
        "\nModels loaded.\n"
    )


    # --------------------------------------------------------
    # YOLO STORAGE
    # --------------------------------------------------------

    yolo_tp = 0
    yolo_fp = 0
    yolo_fn = 0

    per_class_stats = {}


    # --------------------------------------------------------
    # FLORENCE STORAGE
    # --------------------------------------------------------

    bleu_scores = []

    rouge_scores = []

    generated_captions = []

    reference_captions = []


    # --------------------------------------------------------
    # OCR STORAGE
    # --------------------------------------------------------

    raw_cer_scores = []

    raw_wer_scores = []

    normalized_cer_scores = []

    normalized_wer_scores = []


    # --------------------------------------------------------
    # DATASET COUNTERS
    # --------------------------------------------------------

    total_frames = 0

    evaluated_frames = 0


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "NETRA EVALUATION REPORT"
    )

    report_lines.append(
        "=" * 70
    )


    # ========================================================
    # PROCESS VIDEOS
    # ========================================================

    for (
        video_name,
        video_data
    ) in GROUND_TRUTH.items():


        video_path = os.path.join(
            UPLOADS_DIR,
            video_name
        )


        print(
            "\n" + "-" * 70
        )

        print(
            f"VIDEO: {video_name}"
        )

        print(
            "-" * 70
        )


        report_lines.append(
            ""
        )

        report_lines.append(
            "-" * 70
        )

        report_lines.append(
            f"VIDEO: {video_name}"
        )

        report_lines.append(
            "Scene: "
            + video_data.get(
                "scene_type",
                "Unknown"
            )
        )


        if not os.path.exists(
            video_path
        ):

            print(
                f"WARNING: Video not found: "
                f"{video_path}"
            )

            continue


        # ====================================================
        # PROCESS ANNOTATED FRAMES
        # ====================================================

        for (
            timestamp,
            gt_data
        ) in video_data[
            "frames"
        ].items():


            total_frames += 1


            print(
                f"\nTimestamp: {timestamp}s"
            )


            frame = (
                extract_frame_at_timestamp(
                    video_path,
                    timestamp
                )
            )


            if frame is None:

                print(
                    "Could not extract frame."
                )

                continue


            evaluated_frames += 1


            # ==================================================
            # YOLO
            # ==================================================

            detections = (
                yolo_model.detect(
                    frame
                )
            )


            detected_objects = [

                detection["name"]

                for detection
                in detections
            ]


            gt_objects = (
                gt_data[
                    "objects"
                ]
            )


            detected_counts = Counter(
                detected_objects
            )


            ground_truth_counts = Counter(
                gt_objects
            )


            all_classes = set(
                detected_counts.keys()
            ).union(
                ground_truth_counts.keys()
            )


            frame_tp = 0
            frame_fp = 0
            frame_fn = 0


            for class_name in all_classes:


                detected_count = (
                    detected_counts.get(
                        class_name,
                        0
                    )
                )


                ground_truth_count = (
                    ground_truth_counts.get(
                        class_name,
                        0
                    )
                )


                tp = min(
                    detected_count,
                    ground_truth_count
                )


                fp = max(
                    0,
                    detected_count
                    - ground_truth_count
                )


                fn = max(
                    0,
                    ground_truth_count
                    - detected_count
                )


                frame_tp += tp
                frame_fp += fp
                frame_fn += fn


                if (
                    class_name
                    not in per_class_stats
                ):

                    per_class_stats[
                        class_name
                    ] = {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0
                    }


                per_class_stats[
                    class_name
                ]["tp"] += tp


                per_class_stats[
                    class_name
                ]["fp"] += fp


                per_class_stats[
                    class_name
                ]["fn"] += fn


            yolo_tp += frame_tp
            yolo_fp += frame_fp
            yolo_fn += frame_fn


            (
                frame_precision,
                frame_recall,
                frame_f1
            ) = calc_precision_recall_f1(

                frame_tp,
                frame_fp,
                frame_fn
            )


            print(
                "\nYOLOv8"
            )

            print(
                f"Ground Truth : "
                f"{gt_objects}"
            )

            print(
                f"Detected     : "
                f"{detected_objects}"
            )

            print(
                f"Precision    : "
                f"{frame_precision:.3f}"
            )

            print(
                f"Recall       : "
                f"{frame_recall:.3f}"
            )

            print(
                f"F1           : "
                f"{frame_f1:.3f}"
            )


            # ==================================================
            # FLORENCE-2
            # ==================================================

            raw_caption = (
                florence_model.caption(
                    frame
                )
            )


            generated_caption = (
                clean_caption(
                    raw_caption
                )
            )


            reference_caption = (
                gt_data[
                    "expected_caption"
                ]
            )


            bleu_score = (
                calculate_bleu(
                    generated_caption,
                    reference_caption
                )
            )


            rouge_l_score = (
                calculate_rouge_l(
                    generated_caption,
                    reference_caption
                )
            )


            bleu_scores.append(
                bleu_score
            )


            rouge_scores.append(
                rouge_l_score
            )


            generated_captions.append(
                generated_caption
            )


            reference_captions.append(
                reference_caption
            )


            print(
                "\nFlorence-2"
            )


            print(
                f"Expected : "
                f"{reference_caption}"
            )


            print(
                f"Generated: "
                f"{generated_caption}"
            )


            print(
                f"BLEU     : "
                f"{bleu_score:.3f}"
            )


            print(
                f"ROUGE-L  : "
                f"{rouge_l_score:.3f}"
            )


            # ==================================================
            # PADDLE OCR
            # ==================================================

            detected_texts = (
                ocr_model.extract_text(
                    frame
                )
            )


            expected_texts = (
                gt_data[
                    "expected_text"
                ]
            )


            ocr_result = (
                calculate_ocr_metrics(
                    detected_texts,
                    expected_texts
                )
            )


            raw_cer_scores.append(
                ocr_result[
                    "raw_cer"
                ]
            )


            raw_wer_scores.append(
                ocr_result[
                    "raw_wer"
                ]
            )


            normalized_cer_scores.append(
                ocr_result[
                    "normalized_cer"
                ]
            )


            normalized_wer_scores.append(
                ocr_result[
                    "normalized_wer"
                ]
            )


            print(
                "\nPaddleOCR"
            )


            print(
                f"Expected Regions : "
                f"{expected_texts}"
            )


            print(
                f"Detected Regions : "
                f"{detected_texts}"
            )


            print(
                "\nRaw Comparison"
            )


            print(
                f"Expected : "
                f"{ocr_result['raw_expected']}"
            )


            print(
                f"Detected : "
                f"{ocr_result['raw_detected']}"
            )


            print(
                f"Raw CER  : "
                f"{ocr_result['raw_cer']:.3f}"
            )


            print(
                f"Raw WER  : "
                f"{ocr_result['raw_wer']:.3f}"
            )


            print(
                "\nNormalized "
                "Order-Invariant Comparison"
            )


            print(
                f"Expected : "
                f"{ocr_result['normalized_expected']}"
            )


            print(
                f"Detected : "
                f"{ocr_result['normalized_detected']}"
            )


            print(
                f"CER      : "
                f"{ocr_result['normalized_cer']:.3f}"
            )


            print(
                f"WER      : "
                f"{ocr_result['normalized_wer']:.3f}"
            )


    # ========================================================
    # FINAL YOLO METRICS
    # ========================================================

    (
        yolo_precision,
        yolo_recall,
        yolo_f1
    ) = calc_precision_recall_f1(

        yolo_tp,
        yolo_fp,
        yolo_fn
    )


    # ========================================================
    # FINAL FLORENCE METRICS
    # ========================================================

    average_bleu = (

        sum(
            bleu_scores
        )
        / len(
            bleu_scores
        )

        if bleu_scores

        else 0.0
    )


    average_rouge_l = (

        sum(
            rouge_scores
        )
        / len(
            rouge_scores
        )

        if rouge_scores

        else 0.0
    )


    # --------------------------------------------------------
    # BERTSCORE
    # --------------------------------------------------------

    average_bert_precision = 0.0

    average_bert_recall = 0.0

    average_bert_f1 = 0.0


    if generated_captions:

        print(
            "\nCalculating BERTScore..."
        )


        P, R, F1 = bert_score(

            generated_captions,

            reference_captions,

            lang="en",

            verbose=True
        )


        average_bert_precision = (
            P.mean().item()
        )


        average_bert_recall = (
            R.mean().item()
        )


        average_bert_f1 = (
            F1.mean().item()
        )


    # ========================================================
    # FINAL OCR METRICS
    # ========================================================

    average_raw_cer = (

        sum(
            raw_cer_scores
        )
        / len(
            raw_cer_scores
        )

        if raw_cer_scores

        else 0.0
    )


    average_raw_wer = (

        sum(
            raw_wer_scores
        )
        / len(
            raw_wer_scores
        )

        if raw_wer_scores

        else 0.0
    )


    average_normalized_cer = (

        sum(
            normalized_cer_scores
        )
        / len(
            normalized_cer_scores
        )

        if normalized_cer_scores

        else 0.0
    )


    average_normalized_wer = (

        sum(
            normalized_wer_scores
        )
        / len(
            normalized_wer_scores
        )

        if normalized_wer_scores

        else 0.0
    )


    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "FINAL NETRA EVALUATION RESULTS"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print(
        "\nYOLOv8 Object Detection"
    )

    print(
        "-" * 40
    )


    print(
        f"Precision : "
        f"{yolo_precision:.3f} "
        f"({yolo_precision * 100:.1f}%)"
    )


    print(
        f"Recall    : "
        f"{yolo_recall:.3f} "
        f"({yolo_recall * 100:.1f}%)"
    )


    print(
        f"F1 Score  : "
        f"{yolo_f1:.3f} "
        f"({yolo_f1 * 100:.1f}%)"
    )


    print(
        "\nNOTE:"
    )

    print(
        "The YOLO values above use "
        "class-count matching."
    )

    print(
        "Use the separate Ultralytics "
        "bounding-box validation result "
        "for mAP."
    )


    # --------------------------------------------------------
    # FLORENCE
    # --------------------------------------------------------

    print(
        "\nFlorence-2 Scene Captioning"
    )

    print(
        "-" * 40
    )


    print(
        f"BLEU               : "
        f"{average_bleu:.3f}"
    )


    print(
        f"ROUGE-L            : "
        f"{average_rouge_l:.3f}"
    )


    print(
        f"BERTScore Precision: "
        f"{average_bert_precision:.3f}"
    )


    print(
        f"BERTScore Recall   : "
        f"{average_bert_recall:.3f}"
    )


    print(
        f"BERTScore F1       : "
        f"{average_bert_f1:.3f}"
    )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print(
        "\nPaddleOCR"
    )

    print(
        "-" * 40
    )


    print(
        "Raw OCR Metrics"
    )


    print(
        f"Raw CER : "
        f"{average_raw_cer:.3f}"
    )


    print(
        f"Raw WER : "
        f"{average_raw_wer:.3f}"
    )


    print(
        "\nNormalized Order-Invariant OCR Metrics"
    )


    print(
        f"Normalized CER : "
        f"{average_normalized_cer:.3f}"
    )


    print(
        f"Normalized WER : "
        f"{average_normalized_wer:.3f}"
    )


    print(
        "\nNOTE: Lower CER/WER values are better."
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "\nDataset"
    )

    print(
        "-" * 40
    )


    print(
        f"Annotated Frames: "
        f"{total_frames}"
    )


    print(
        f"Evaluated Frames: "
        f"{evaluated_frames}"
    )


    # ========================================================
    # WRITE REPORT
    # ========================================================

    report_lines.append(
        ""
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "FINAL NETRA EVALUATION RESULTS"
    )

    report_lines.append(
        "=" * 70
    )


    report_lines.append(
        ""
    )

    report_lines.append(
        "YOLOv8 Object Detection"
    )


    report_lines.append(
        f"Precision: "
        f"{yolo_precision:.3f}"
    )


    report_lines.append(
        f"Recall: "
        f"{yolo_recall:.3f}"
    )


    report_lines.append(
        f"F1 Score: "
        f"{yolo_f1:.3f}"
    )


    report_lines.append(
        "NOTE: YOLO values above use "
        "class-count matching."
    )


    report_lines.append(
        "Use separate bounding-box "
        "validation output for mAP."
    )


    # --------------------------------------------------------
    # FLORENCE REPORT
    # --------------------------------------------------------

    report_lines.append(
        ""
    )

    report_lines.append(
        "Florence-2 Scene Captioning"
    )


    report_lines.append(
        f"BLEU: "
        f"{average_bleu:.3f}"
    )


    report_lines.append(
        f"ROUGE-L: "
        f"{average_rouge_l:.3f}"
    )


    report_lines.append(
        f"BERTScore Precision: "
        f"{average_bert_precision:.3f}"
    )


    report_lines.append(
        f"BERTScore Recall: "
        f"{average_bert_recall:.3f}"
    )


    report_lines.append(
        f"BERTScore F1: "
        f"{average_bert_f1:.3f}"
    )


    # --------------------------------------------------------
    # OCR REPORT
    # --------------------------------------------------------

    report_lines.append(
        ""
    )

    report_lines.append(
        "PaddleOCR"
    )


    report_lines.append(
        f"Raw CER: "
        f"{average_raw_cer:.3f}"
    )


    report_lines.append(
        f"Raw WER: "
        f"{average_raw_wer:.3f}"
    )


    report_lines.append(
        f"Normalized CER: "
        f"{average_normalized_cer:.3f}"
    )


    report_lines.append(
        f"Normalized WER: "
        f"{average_normalized_wer:.3f}"
    )


    report_lines.append(
        "Lower CER/WER is better."
    )


    # --------------------------------------------------------
    # DATASET REPORT
    # --------------------------------------------------------

    report_lines.append(
        ""
    )


    report_lines.append(
        f"Annotated Frames: "
        f"{total_frames}"
    )


    report_lines.append(
        f"Evaluated Frames: "
        f"{evaluated_frames}"
    )


    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )


    print(
        f"\nEvaluation report saved to:\n"
        f"{REPORT_FILE}"
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    try:
        yolo_model.unload()
    except Exception:
        pass


    try:
        ocr_model.unload()
    except Exception:
        pass


    try:
        florence_model.unload()
    except Exception:
        pass


    print(
        "\nEvaluation complete."
    )


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()