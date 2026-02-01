"""
Image preprocessing for font detection.

Cleans up real-world screenshots and photos so they work well with the
OpenCLIP model trained on clean synthetic samples.

Two paths:
  - preprocess_single(): For pre-cropped regions (Chrome extension use case)
  - preprocess_for_model(): For full photos with multiple text regions (upload use case)
"""

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms


class FontImagePreprocessor:
    """Prepares real-world font images for the OpenCLIP embedding model."""

    # OpenCLIP ViT-B-32 normalization constants
    OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        # Only ToTensor + Normalize — we handle resize/pad ourselves to
        # avoid OpenCLIP's default CenterCrop which cuts off wide text.
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.OPENCLIP_MEAN, std=self.OPENCLIP_STD),
        ])
        self._ocr_reader = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_single(self, image: Image.Image, preprocess_fn=None) -> torch.Tensor:
        """Primary path for the Chrome extension.

        The user already selected a region, so skip text detection.
        Pipeline: deskew -> normalize background -> resize+pad -> tensor normalize.

        Args:
            image: PIL Image (RGB).
            preprocess_fn: Ignored (kept for interface symmetry).

        Returns:
            Tensor of shape ``[1, 3, 224, 224]``.
        """
        image = image.convert("RGB")
        image = self.deskew(image)
        image = self.normalize_background(image)
        image = self.resize_and_pad(image, self.target_size)
        tensor = self.tensor_transform(image)
        return tensor.unsqueeze(0)

    def preprocess_for_model(self, image: Image.Image, preprocess_fn=None) -> torch.Tensor:
        """Secondary path for uploaded photos with multiple text regions.

        Runs EasyOCR text detection, crops each detected region, then runs
        the single-region pipeline on each crop.

        Args:
            image: PIL Image (RGB).
            preprocess_fn: Ignored.

        Returns:
            Tensor of shape ``[N, 3, 224, 224]`` where N is the number of
            detected text regions (at least 1 — falls back to full image).
        """
        image = image.convert("RGB")
        boxes = self._detect_text_regions(image)

        if not boxes:
            # No text detected — treat the whole image as one region
            return self.preprocess_single(image)

        tensors = []
        for (x1, y1, x2, y2) in boxes:
            crop = image.crop((x1, y1, x2, y2))
            tensors.append(self.preprocess_single(crop))

        return torch.cat(tensors, dim=0)

    # ------------------------------------------------------------------
    # Image cleanup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def deskew(image: Image.Image) -> Image.Image:
        """Straighten slightly rotated text using OpenCV contour analysis."""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Use the largest contour to determine skew angle
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[2]

        # minAreaRect returns angles in [-90, 0); normalise to [-45, 45]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        # Only correct small skews (< 15 degrees) to avoid mangling
        if abs(angle) < 0.5 or abs(angle) > 15:
            return image

        h, w = img_cv.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img_cv, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    @staticmethod
    def normalize_background(image: Image.Image) -> Image.Image:
        """Ensure dark-text-on-light-background and boost contrast."""
        gray = image.convert("L")
        mean_brightness = np.array(gray).mean()

        # If the image is predominantly dark, it's likely light text on dark bg
        if mean_brightness < 128:
            image = ImageOps.invert(image)

        image = ImageOps.autocontrast(image)
        return image

    @staticmethod
    def resize_and_pad(image: Image.Image, target: int = 224) -> Image.Image:
        """Resize longest side to *target*, paste centred on white canvas.

        This avoids OpenCLIP's default CenterCrop which cuts off text on
        wide or tall images.
        """
        w, h = image.size
        scale = target / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        paste_x = (target - new_w) // 2
        paste_y = (target - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        return canvas

    # ------------------------------------------------------------------
    # Text detection (lazy-loaded EasyOCR)
    # ------------------------------------------------------------------

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        return self._ocr_reader

    def _detect_text_regions(self, image: Image.Image):
        """Return a list of (x1, y1, x2, y2) bounding boxes for text regions."""
        reader = self._get_ocr_reader()
        img_np = np.array(image)
        results = reader.readtext(img_np)

        boxes = []
        for (bbox, _text, _conf) in results:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            # Skip tiny detections
            if (x2 - x1) > 10 and (y2 - y1) > 5:
                boxes.append((x1, y1, x2, y2))

        return boxes
