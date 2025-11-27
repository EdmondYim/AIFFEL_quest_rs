"""
EasyOCR(탐지) + 학습한 CRNN(인식)으로 FUNSD를 평가하는 스크립트.
FUNSD 학습 데이터의 GT 박스를 활용해 정답과 비교할 수 있다.
"""

import glob
import json
from pathlib import Path

import easyocr
import numpy as np
import torch
from PIL import Image, ImageDraw

from dataloader import TARGET_CHARACTERS, LabelConverter
from inference import decode_greedy
from train import CRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 탐지는 EasyOCR, 인식은 학습된 CRNN 사용
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
label_converter = LabelConverter(TARGET_CHARACTERS)
num_chars = len(label_converter.character) + 1
model = CRNN(num_chars=num_chars).to(device)
model.load_state_dict(torch.load("./checkpoints/model_checkpoint.pth", map_location=device))
model.eval()


def preprocess_crop(pil_img, target_size=(100, 32)):
    """오른쪽 패딩으로 리사이즈 후 (1,3,H,W)로 정규화."""
    img = np.array(pil_img.convert("RGB"))
    h, w, _ = img.shape
    new_w = min(int(w * target_size[1] / h), target_size[0])

    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    resized = np.array(Image.fromarray(img).resize((new_w, target_size[1])))
    canvas[:, :new_w, :] = resized

    canvas = canvas.transpose(2, 0, 1) / 255.0  # (3,H,W), [0,1]
    return torch.tensor(canvas, dtype=torch.float32).unsqueeze(0)  # (1,3,H,W)


def detect_and_recognize(img_path):
    """EasyOCR로 탐지 후 crop을 CRNN으로 인식."""
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    detections = reader.readtext(img_np, detail=1, paragraph=False)  # bbox, text, conf

    img_draw = ImageDraw.Draw(img_pil)
    texts = []

    with torch.no_grad():
        for bbox, _, _ in detections:
            bbox_int = [(int(x), int(y)) for x, y in bbox]
            img_draw.polygon(bbox_int, outline="red")

            xs, ys = zip(*bbox_int)
            crop = img_pil.crop(
                (max(0, min(xs) - 5), max(0, min(ys) - 5), max(xs) + 5, max(ys) + 5)
            )

            x = preprocess_crop(crop).to(device)
            logits = model(x)  # (T,1,C)
            pred = decode_greedy(logits, label_converter)[0]
            texts.append(pred)

    return img_pil, texts


def pick_sample_from_funsd(split="training_data"):
    """FUNSD split(training_data/testing_data)에서 첫 이미지를 고른다."""
    candidates = glob.glob(f"dataset/dataset/{split}/images/*.png")
    if not candidates:
        raise FileNotFoundError(f"FUNSD images not found under dataset/dataset/{split}/images")
    return candidates[0]


def load_funsd_annotation(image_path: Path):
    """FUNSD 이미지에 매칭되는 JSON 어노테이션을 읽는다."""
    ann_path = image_path.parent.parent / "annotations" / f"{image_path.stem}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")
    return json.loads(ann_path.read_text(encoding="utf-8"))


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def annotate_results(img, results, out_path, max_draw=100):
    """bbox에 pred|gt를 그린다(정답=초록, 오답=빨강)."""
    draw = ImageDraw.Draw(img)
    for r in results[:max_draw]:
        box = r["box"]
        color = "green" if r["ed"] == 0 else "red"
        draw.rectangle(box, outline=color, width=2)
        pred_txt = r.get("pred_raw", r["pred"])
        gt_txt = r.get("gt_raw", r["gt"])
        text = f"{pred_txt}|{gt_txt}"
        draw.text((box[0], max(0, box[1] - 12)), text, fill=color)
    img.save(out_path)


def normalize_text(text, case_insensitive=True):
    return text.upper() if case_insensitive else text


def evaluate_on_funsd_training(
    max_words=50, top_n_print=10, case_insensitive=True, img_path=None, annotate_prefix="annotated_funsd_training"
):
    """FUNSD 학습 데이터의 GT 박스로 crop을 잘라 예측과 비교."""
    img_path = Path(img_path) if img_path else Path(pick_sample_from_funsd(split="training_data"))
    ann = load_funsd_annotation(img_path)
    img = Image.open(img_path).convert("RGB")

    words = []
    for block in ann.get("form", []):
        for w in block.get("words", []):
            text_raw = w.get("text", "").strip()
            if text_raw:
                words.append((w["box"], text_raw))

    results = []
    with torch.no_grad():
        for idx, (box, gt) in enumerate(words[:max_words]):
            x_min, y_min, x_max, y_max = box
            crop = img.crop((x_min, y_min, x_max, y_max))
            x = preprocess_crop(crop).to(device)
            logits = model(x)
            pred_raw = decode_greedy(logits, label_converter)[0]
            gt_norm = normalize_text(gt, case_insensitive=case_insensitive)
            pred_norm = normalize_text(pred_raw, case_insensitive=case_insensitive)
            ed = levenshtein(gt_norm, pred_norm)
            results.append(
                {
                    "gt": gt_norm,
                    "pred": pred_norm,
                    "gt_raw": gt,
                    "pred_raw": pred_raw,
                    "ed": ed,
                    "box": (x_min, y_min, x_max, y_max),
                }
            )

    total = len(results)
    correct = sum(1 for r in results if r["ed"] == 0)
    total_ed = sum(r["ed"] for r in results)

    print(f"Image: {img_path.name}")
    print(f"Evaluated words: {total} (max_words={max_words})")
    print(f"Exact matches: {correct} ({correct/total*100:.2f}%)")
    print(f"Average edit distance: {total_ed/total:.2f}")
    print(f"Case-insensitive eval: {case_insensitive}")
    print("\nTop samples by edit distance (gt -> pred, ed):")
    results_sorted = sorted(results, key=lambda r: (-r["ed"], -len(r["gt"])))
    for r in results_sorted[:top_n_print]:
        print(f"{r['gt_raw']} -> {r['pred_raw']} (ed={r['ed']})")

    # 시각화 저장
    out_path = Path(f"{annotate_prefix}_{img_path.stem}.png")
    annotate_results(img.copy(), results_sorted, out_path, max_draw=min(100, len(results_sorted)))
    print(f"\nAnnotated image saved to: {out_path}")


def evaluate_multiple_from_funsd(num_images=10, **kwargs):
    """FUNSD 학습 이미지를 여러 장 순회하며 GT 박스로 평가."""
    candidates = sorted(glob.glob("dataset/dataset/training_data/images/*.png"))
    if not candidates:
        raise FileNotFoundError(
            "FUNSD training_data images not found under dataset/dataset/training_data/images"
        )

    for img_path in candidates[:num_images]:
        evaluate_on_funsd_training(img_path=img_path, **kwargs)
        print("-" * 60)


if __name__ == "__main__":
    # GT 박스로 평가
    evaluate_multiple_from_funsd(num_images=10, max_words=100, top_n_print=10, case_insensitive=True)
