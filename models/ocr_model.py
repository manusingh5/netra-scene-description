# models/ocr_model.py
"""
NETRA PaddleOCR Text Extraction Wrapper
REAL MODEL — Using PaddleOCR for text extraction
"""

from typing import List, Dict

class OCRModel:
    """
    PaddleOCR text extraction wrapper
    Real model loaded on initialization (~500MB)
    """

    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.50
        print("[OCR] Loading PaddleOCR...")
        
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR (English language, use GPU if available)
            
            self.model = PaddleOCR(
                lang='en',           # English language
                use_angle_cls=True,  # Text direction detection
                det_limit_side_len=960,  # Image resize limit
            )


            print("[OCR] PaddleOCR loaded successfully! ✅")
        except ImportError:
            print("[OCR ERROR] paddleocr not found. Install with: pip install paddleocr")
        except Exception as e:
            print(f"[OCR ERROR] Failed to load model: {e}")

    def extract_text(self, frame) -> List[str]:
        """
        Extract text from a frame using PaddleOCR

        Args:
            frame: numpy array (BGR image) from OpenCV

        Returns:
            List of detected text strings
        """
        if self.model is None:
            print("[OCR WARNING] Model not loaded. Returning empty text.")
            return []

        try:
            # Run OCR inference
            result = self.model.ocr(frame, cls=True)
            
            # Parse results
            texts = []
            if result and result[0]:  # Result structure: [[[(bbox, text_conf), ...], ...]]
                for line in result[0]:
                    if len(line) >= 2:
                        text = line[1][0]  # Text string
                        confidence = line[1][1]  # Confidence score
                        
                        if confidence >= self.confidence_threshold:
                            texts.append(text)
            
            return texts

        except Exception as e:
            print(f"[OCR ERROR] Inference failed: {e}")
            return []

    def unload(self):
        """Free model memory"""
        if self.model:
            # PaddleOCR cleanup
            del self.model
            self.model = None
            print("[OCR] Model unloaded")