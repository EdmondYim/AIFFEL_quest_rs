import numpy as np
import torch
import csv

from dataloader import TARGET_CHARACTERS, LabelConverter, MJDataset
from train import CRNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
data_dir = 'data_lmdb_release/data_lmdb_release/training/MJ'
TEST_DATA_PATH = data_dir + '/MJ_test'

# Hyperparameters
IMG_SIZE = (100, 32)


def decode_greedy(output, label_converter):
    # (T,B,C) -> (B,T) index
    out = output.detach().cpu().numpy()  # (T,B,C)
    argmax = out.argmax(axis=2).transpose()  # (B,T)

    results = []
    for seq in argmax:
        # 연속된 동일 글자(또는 blank=0) 제거 로직을 적용해야
        # CTC 디코딩다운 결과가 나옵니다.
        # 여기서는 간단히 blank(0) 무시하고 연속 제거만 보여줌
        decoded = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:
                decoded.append(idx)
            prev = idx
        # 인덱스를 실제 문자로
        decoded_str = label_converter.decode(decoded)
        results.append(decoded_str)
    return results


def levenshtein_distance(s1, s2):
    """Levenshtein distance (Edit distance) 계산"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer than s2
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def check_inference(model, dataset, label_converter, index=20000, show_images=False):
    results = []

    for i in range(index):
        img, encoded_label, label_len, raw_label = dataset[i]  # 단일 데이터
        img_tensor = torch.tensor(img[np.newaxis, ...], dtype=torch.float32).to(
            device)  # (1,3,32,100)

        # 이미지 정규화
        img_tensor = img_tensor / 255.0

        output = model(img_tensor)  # (T,1,num_chars)
        # 디코딩
        result_text = decode_greedy(output, label_converter)[0]

        # Edit distance 계산
        edit_dist = levenshtein_distance(raw_label, result_text)

        # 결과 저장
        results.append({
            'index': i,
            'ground_truth': raw_label,
            'prediction': result_text,
            'correct': raw_label == result_text,
            'edit_distance': edit_dist,
            'gt_length': len(raw_label),
            'pred_length': len(result_text)
        })

        print(f"GT: {raw_label} / Pred: {result_text}", flush=True)

    return results


if __name__ == "__main__":
    # 학습된 모델의 가중치가 저장된 경로
    checkpoint_path = "./checkpoints/model_checkpoint.pth"

    # Label converter 초기화
    label_converter = LabelConverter(TARGET_CHARACTERS)
    num_chars = len(label_converter.character) + 1  # +1 for blank
    print(f"Number of classes (including blank): {num_chars}")

    # 테스트 데이터셋 생성
    test_dataset = MJDataset(
        TEST_DATA_PATH,
        label_converter=label_converter,
        img_size=IMG_SIZE,
        character=TARGET_CHARACTERS
    )

    # 모델 로드
    model = CRNN(num_chars=num_chars).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 추론 테스트 (no grad)
    num_samples = 25000
    print(f"추론 시작: {num_samples}개 샘플")

    with torch.no_grad():
        results = check_inference(
            model, test_dataset, label_converter, index=num_samples, show_images=False)

    # CSV로 저장
    csv_path = 'inference_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
                                'index', 'ground_truth', 'prediction', 'correct', 'edit_distance', 'gt_length', 'pred_length'])
        writer.writeheader()
        writer.writerows(results)

    # Metrics 계산
    correct_count = sum(1 for r in results if r['correct'])
    total_samples = len(results)
    accuracy = correct_count / total_samples * 100

    # Character Error Rate (CER) 계산
    total_edit_distance = sum(r['edit_distance'] for r in results)
    total_characters = sum(r['gt_length'] for r in results)
    cer = (total_edit_distance / total_characters) * 100

    # 평균 Edit Distance
    avg_edit_distance = total_edit_distance / total_samples

    # 완벽 예측 vs 부분 오류 vs 완전 오류 분석
    perfect = sum(1 for r in results if r['edit_distance'] == 0)
    partial_errors = sum(1 for r in results if 0 <
                         r['edit_distance'] < r['gt_length'])
    complete_errors = sum(
        1 for r in results if r['edit_distance'] >= r['gt_length'])

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Test Dataset Evaluation Metrics")
    print(f"{'='*60}")
    print(f"총 샘플 수: {total_samples}")
    print(f"\n[정확도 (Accuracy)]")
    print(f"  정확한 예측: {correct_count} ({accuracy:.2f}%)")
    print(f"  오류: {total_samples - correct_count} ({100-accuracy:.2f}%)")
    print(f"\n[문자 단위 오류율 (Character Error Rate)]")
    print(f"  CER: {cer:.2f}%")
    print(f"  총 문자 수: {total_characters}")
    print(f"  총 편집 거리: {total_edit_distance}")
    print(f"\n[편집 거리 (Edit Distance)]")
    print(f"  평균 편집 거리: {avg_edit_distance:.2f}")
    print(f"\n[오류 분석]")
    print(f"  완벽 예측 (ED=0): {perfect} ({perfect/total_samples*100:.1f}%)")
    print(
        f"  부분 오류 (0<ED<Len): {partial_errors} ({partial_errors/total_samples*100:.1f}%)")
    print(
        f"  심각한 오류 (ED≥Len): {complete_errors} ({complete_errors/total_samples*100:.1f}%)")
    print(f"\n결과 저장: {csv_path}")
    print(f"{'='*60}")
