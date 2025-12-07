"""
학습 결과 시각화 및 비교 분석 스크립트

사용법:
    python visualize_results.py
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# 체크포인트 경로
CHECKPOINT_DIR = 'checkpoints'


def load_history(model_name):
    """체크포인트에서 학습 히스토리 로드"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}-latest.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('history', None)
    return None


def normalize_loss(history, num_stacks):
    """
    Loss를 정규화하여 공정한 비교가 가능하도록 함
    Hourglass는 num_stacks만큼 loss가 합산되므로 나눠줌
    """
    if history is None:
        return None
    
    normalized = history.copy()
    normalized['train_loss'] = [l / num_stacks for l in history['train_loss']]
    normalized['val_loss'] = [l / num_stacks for l in history['val_loss']]
    return normalized


def plot_training_comparison(history_hg, history_sb, save_path='training_comparison.png'):
    """
    두 모델의 학습 결과를 2x2 그리드로 시각화
    """
    # Loss 정규화 (Hourglass는 4-stack이므로 4로 나눔)
    history_hg_norm = normalize_loss(history_hg, num_stacks=4)
    
    epochs_hg = range(1, len(history_hg['train_loss']) + 1)
    epochs_sb = range(1, len(history_sb['train_loss']) + 1)
    
    # 스타일 설정
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    
    colors = {
        'hg_train': '#2E86AB',  # Blue
        'hg_val': '#A23B72',    # Pink
        'sb_train': '#F18F01',  # Orange
        'sb_val': '#C73E1D'     # Red
    }
    
    # ========================================
    # 1. Raw Loss Comparison (왼쪽 상단)
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_hg, history_hg['train_loss'], 'o-', color=colors['hg_train'], 
             label='Hourglass Train', linewidth=2, markersize=6)
    ax1.plot(epochs_hg, history_hg['val_loss'], 's--', color=colors['hg_val'], 
             label='Hourglass Val', linewidth=2, markersize=6)
    ax1.plot(epochs_sb, history_sb['train_loss'], 'o-', color=colors['sb_train'], 
             label='SimpleBaseline Train', linewidth=2, markersize=6)
    ax1.plot(epochs_sb, history_sb['val_loss'], 's--', color=colors['sb_val'], 
             label='SimpleBaseline Val', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Raw Loss (Not Comparable)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, max(len(epochs_hg), len(epochs_sb)) + 1))
    
    # ========================================
    # 2. Normalized Loss Comparison (오른쪽 상단)
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_hg, history_hg_norm['train_loss'], 'o-', color=colors['hg_train'], 
             label='Hourglass Train', linewidth=2, markersize=6)
    ax2.plot(epochs_hg, history_hg_norm['val_loss'], 's--', color=colors['hg_val'], 
             label='Hourglass Val', linewidth=2, markersize=6)
    ax2.plot(epochs_sb, history_sb['train_loss'], 'o-', color=colors['sb_train'], 
             label='SimpleBaseline Train', linewidth=2, markersize=6)
    ax2.plot(epochs_sb, history_sb['val_loss'], 's--', color=colors['sb_val'], 
             label='SimpleBaseline Val', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Normalized Loss', fontsize=11)
    ax2.set_title('Normalized Loss (Comparable)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, max(len(epochs_hg), len(epochs_sb)) + 1))
    
    # ========================================
    # 3. Accuracy Comparison (왼쪽 하단)
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs_hg, history_hg['train_acc'], 'o-', color=colors['hg_train'], 
             label='Hourglass Train', linewidth=2, markersize=6)
    ax3.plot(epochs_hg, history_hg['val_acc'], 's--', color=colors['hg_val'], 
             label='Hourglass Val', linewidth=2, markersize=6)
    ax3.plot(epochs_sb, history_sb['train_acc'], 'o-', color=colors['sb_train'], 
             label='SimpleBaseline Train', linewidth=2, markersize=6)
    ax3.plot(epochs_sb, history_sb['val_acc'], 's--', color=colors['sb_val'], 
             label='SimpleBaseline Val', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy (PCKh)', fontsize=11)
    ax3.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, max(len(epochs_hg), len(epochs_sb)) + 1))
    ax3.set_ylim(0, 1)
    
    # ========================================
    # 4. Final Results Summary (오른쪽 하단)
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # 최종 결과 표
    final_hg_loss = history_hg_norm['val_loss'][-1]
    final_sb_loss = history_sb['val_loss'][-1]
    final_hg_acc = history_hg['val_acc'][-1]
    final_sb_acc = history_sb['val_acc'][-1]
    
    summary_text = """
    ╔══════════════════════════════════════════════════╗
    ║           FINAL RESULTS SUMMARY                  ║
    ╠══════════════════════════════════════════════════╣
    ║                                                  ║
    ║  Metric             Hourglass    SimpleBaseline  ║
    ║  ────────────────────────────────────────────    ║
    ║  Normalized Loss    {:.4f}         {:.4f}        ║
    ║  Accuracy (PCKh)    {:.4f}         {:.4f}        ║
    ║                                                  ║
    ╠══════════════════════════════════════════════════╣
    ║  WINNER: {}                           ║
    ╚══════════════════════════════════════════════════╝
    """.format(
        final_hg_loss, final_sb_loss,
        final_hg_acc, final_sb_acc,
        "Hourglass (Higher Accuracy)    " if final_hg_acc > final_sb_acc else "SimpleBaseline (Faster)        "
    )
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', 
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))
    
    plt.suptitle('Hourglass vs SimpleBaseline Training Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")
    plt.show()


def plot_per_epoch_details(history_hg, history_sb, save_path='per_epoch_details.png'):
    """
    에폭별 상세 정보 시각화
    """
    history_hg_norm = normalize_loss(history_hg, num_stacks=4)
    
    epochs = range(1, len(history_hg['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart for final epoch comparison
    x = np.arange(2)
    width = 0.35
    
    # Accuracy comparison
    ax1 = axes[0]
    final_accs = [history_hg['val_acc'][-1], history_sb['val_acc'][-1]]
    bars1 = ax1.bar(x, final_accs, width, color=['#2E86AB', '#F18F01'], edgecolor='black')
    ax1.set_ylabel('Validation Accuracy', fontsize=11)
    ax1.set_title('Final Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Hourglass', 'SimpleBaseline'])
    ax1.set_ylim(0, 1)
    ax1.bar_label(bars1, fmt='%.3f', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Loss comparison (normalized)
    ax2 = axes[1]
    final_losses = [history_hg_norm['val_loss'][-1], history_sb['val_loss'][-1]]
    bars2 = ax2.bar(x, final_losses, width, color=['#2E86AB', '#F18F01'], edgecolor='black')
    ax2.set_ylabel('Normalized Validation Loss', fontsize=11)
    ax2.set_title('Final Loss Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Hourglass', 'SimpleBaseline'])
    ax2.bar_label(bars2, fmt='%.4f', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")
    plt.show()


def print_detailed_summary(history_hg, history_sb):
    """상세 요약 출력"""
    history_hg_norm = normalize_loss(history_hg, num_stacks=4)
    
    print("\n" + "="*70)
    print("                    DETAILED TRAINING SUMMARY")
    print("="*70)
    
    print("\n[Epoch-by-Epoch Results]")
    print("-"*70)
    print(f"{'Epoch':<8} {'HG Train Loss':>14} {'HG Val Loss':>12} {'HG Val Acc':>11} │ "
          f"{'SB Train Loss':>14} {'SB Val Loss':>12} {'SB Val Acc':>11}")
    print("-"*70)
    
    for i in range(len(history_hg['train_loss'])):
        print(f"{i+1:<8} {history_hg_norm['train_loss'][i]:>14.4f} {history_hg_norm['val_loss'][i]:>12.4f} "
              f"{history_hg['val_acc'][i]:>11.4f} │ "
              f"{history_sb['train_loss'][i]:>14.4f} {history_sb['val_loss'][i]:>12.4f} "
              f"{history_sb['val_acc'][i]:>11.4f}")
    
    print("-"*70)
    
    # 최종 비교
    print("\n[Final Comparison (Normalized)]")
    print("-"*40)
    
    final_hg_loss = history_hg_norm['val_loss'][-1]
    final_sb_loss = history_sb['val_loss'][-1]
    final_hg_acc = history_hg['val_acc'][-1]
    final_sb_acc = history_sb['val_acc'][-1]
    
    loss_diff = ((final_sb_loss - final_hg_loss) / final_sb_loss) * 100
    acc_diff = ((final_hg_acc - final_sb_acc) / final_sb_acc) * 100
    
    print(f"Hourglass Val Loss (normalized): {final_hg_loss:.4f}")
    print(f"SimpleBaseline Val Loss:         {final_sb_loss:.4f}")
    print(f"  → Hourglass is {abs(loss_diff):.1f}% {'lower' if loss_diff > 0 else 'higher'} loss")
    print()
    print(f"Hourglass Val Accuracy:          {final_hg_acc:.4f}")
    print(f"SimpleBaseline Val Accuracy:     {final_sb_acc:.4f}")
    print(f"  → Hourglass is {abs(acc_diff):.1f}% {'higher' if acc_diff > 0 else 'lower'} accuracy")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    if final_hg_acc > final_sb_acc:
        print("  ✓ Hourglass achieved BETTER accuracy")
        print(f"    ({final_hg_acc:.4f} vs {final_sb_acc:.4f}, +{acc_diff:.1f}%)")
    else:
        print("  ✓ SimpleBaseline achieved BETTER accuracy")
        print(f"    ({final_sb_acc:.4f} vs {final_hg_acc:.4f})")
    print("="*70 + "\n")


def main():
    print("Loading training histories...")
    
    # 히스토리 로드
    history_hg = load_history('hourglass')
    history_sb = load_history('simplebaseline')
    
    if history_hg is None:
        print("Error: Hourglass checkpoint not found!")
        return
    if history_sb is None:
        print("Error: SimpleBaseline checkpoint not found!")
        return
    
    print(f"Loaded Hourglass: {len(history_hg['train_loss'])} epochs")
    print(f"Loaded SimpleBaseline: {len(history_sb['train_loss'])} epochs")
    
    # 상세 요약 출력
    print_detailed_summary(history_hg, history_sb)
    
    # 시각화
    plot_training_comparison(history_hg, history_sb)
    plot_per_epoch_details(history_hg, history_sb)


if __name__ == "__main__":
    main()
