import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imread

from models import UNet, UNetPlusPlus
from visualize_augmentation import build_augmentation
from metrics import compute_iou


def create_palette(num_classes):
    """Deterministic color map (Pascal-style bit trick) for N classes."""
    palette = []
    for cls in range(num_classes):
        label = cls
        r = g = b = 0
        for i in range(8):
            r |= ((label >> 0) & 1) << (7 - i)
            g |= ((label >> 1) & 1) << (7 - i)
            b |= ((label >> 2) & 1) << (7 - i)
            label >>= 3
        palette.append((r, g, b))
    return palette


def decode_segmap(mask, palette):
    """Convert class index mask to RGB using provided palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        rgb[mask == cls] = color
    return rgb


def get_output(model, preproc, image_path, output_path, label_path=None, device="cuda", palette=None):
    """Run inference on a single image and save overlay + color map."""
    palette = palette or create_palette(model.num_classes)

    origin_img = imread(image_path)
    data = {"image": origin_img}
    processed = preproc(**data)

    input_tensor = torch.tensor(
        processed["image"] / 255.0, dtype=torch.float32)
    input_tensor = input_tensor.permute(
        2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, list):  # deep supervision
            output = output[-1]

    prediction = torch.argmax(
        output, dim=1).squeeze().cpu().numpy().astype(np.int64)
    rgb_prediction = decode_segmap(prediction, palette)
    pred_img = Image.fromarray(rgb_prediction).convert("RGBA")
    pred_img_resized = pred_img.resize(
        (origin_img.shape[1], origin_img.shape[0]), Image.NEAREST)

    background = Image.fromarray(origin_img).convert("RGBA")
    blended = Image.blend(background, pred_img_resized, alpha=0.5)
    blended.save(output_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(origin_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb_prediction)
    plt.title("Prediction (Color Map)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title("Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    if label_path and os.path.exists(label_path):
        target = imread(label_path)
        miou, _ = compute_iou(
            prediction[None, ...], target[None, ...], num_classes=output.shape[1], ignore_index=-100)
        print(f"Mean IoU vs label: {miou:.4f}")

    return blended, prediction


def main():
    DATA_DIR = "data_semantics/training"
    MODEL_TYPE = "unet"  # "unet" or "unetpp"
    NUM_CLASSES = 34
    MODEL_PATH = (
        "training/seg_model_unet.pth"
        if MODEL_TYPE == "unet"
        else "training/seg_model_unetplusplus.pth"
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {DEVICE}")
    palette = create_palette(NUM_CLASSES)

    # Choose model to match the checkpoint architecture
    if MODEL_TYPE == "unet":
        model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    else:
        model = UNetPlusPlus(
            in_channels=3, num_classes=NUM_CLASSES, deep_supervision=True
        ).to(DEVICE)

    # Load checkpoint
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f"[Load error] {e}")
            print(
                "Check that MODEL_TYPE matches the checkpoint "
                "and NUM_CLASSES matches the head used in training."
            )
            return
        print(f"Loaded model: {MODEL_PATH} as {MODEL_TYPE}")
    else:
        print(f"Model checkpoint not found: {MODEL_PATH}")
        return

    test_preproc = build_augmentation(is_train=False)

    i = 10
    image_path = os.path.join(DATA_DIR, "image_2", f"{str(i).zfill(6)}_10.png")
    label_path = os.path.join(DATA_DIR, "semantic",
                              f"{str(i).zfill(6)}_10.png")
    output_path = f"result_{str(i).zfill(6)}.png"

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    print(f"Testing on: {image_path}")
    get_output(model, test_preproc, image_path,
               output_path, label_path, DEVICE, palette)


if __name__ == "__main__":
    main()
