import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def load_results(csv_path):  # csv 불러오기
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    results = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                {
                    "ground_truth": row["ground_truth"],
                    "prediction": row["prediction"],
                    "correct": row["correct"].lower() == "true",
                    "edit_distance": int(row["edit_distance"]),
                    "gt_length": int(row["gt_length"]),
                    "pred_length": int(row["pred_length"]),
                }
            )
    return results
# 결과에 gt, pred, correct, edit_distance, gt_length, pred_length 포함


def analyze_confusions(results, top_k=20):
    pair_counter = Counter()
    len_tot = Counter()
    len_correct = Counter()
    len_ed = Counter()

    for r in results:
        gt, pred = r["ground_truth"], r["prediction"]
        length = r["gt_length"]
        len_tot[length] += 1
        if r["correct"]:
            len_correct[length] += 1
        len_ed[(length, r["edit_distance"])] += 1

        if gt == pred:
            continue

        # 얼마나 어느 부분에서 틀렸는지, 어떤 오타가 있었는지 분석, 길이에 따른 통계
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, gt, pred).get_opcodes():
            if tag == "replace":
                for a, b in zip(gt[i1:i2], pred[j1:j2]):
                    pair_counter[(a, b)] += 1
            elif tag == "delete":
                for a in gt[i1:i2]:
                    pair_counter[(a, "<del>")] += 1
            elif tag == "insert":
                for b in pred[j1:j2]:
                    pair_counter[("<ins>", b)] += 1

    confusions = pair_counter.most_common(top_k)
    acc_per_len = {
        L: len_correct[L] / len_tot[L] * 100
        for L in len_tot
        if len_tot[L] > 0
    }
    return confusions, len_tot, len_correct, acc_per_len, len_ed


def main(csv_path="inference_results.csv", top_k=20):
    results = load_results(csv_path)

    (
        confusions,
        len_tot,
        len_correct,
        acc_per_len,
        len_ed,
    ) = analyze_confusions(results, top_k=top_k)

    print("\n[Top confusions]")
    for (g, p), cnt in confusions:
        print(f"{g} -> {p}: {cnt}")

    print("\n[Accuracy by length]")
    for L in sorted(len_tot):
        acc = acc_per_len.get(L, 0.0)
        print(f"len={L}: {len_correct[L]}/{len_tot[L]} ({acc:.2f}%)")

    print("\n[Edit-distance by length] (top counts)")
    for (L, ed), cnt in Counter(len_ed).most_common(20):
        print(f"len={L}, ed={ed}: {cnt}")


if __name__ == "__main__":
    # 수정 가능: csv_path, top_k
    main(csv_path="inference_results.csv", top_k=20)
