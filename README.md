# GNAI-Core

**GNAI-Core (Minimal Class)**

JAXベースの最小コア：Graph Attention + Free Energy Principle + Heun Integrator を統合した几何学的認知・感情シミュレーション研究用フレームワーク。

## 特徴

- Graph Neural Attention with geometric Laplacian (W_geo, softplus symmetrized)
- Free Energy Principle inspired energy terms (F_err with Huber + precision, F_exp with residue target, F_trn transition, F_geo geometry kernel, F_reg)
- Heun integrator for stable simultaneous state (x, mu, r) and params update
- Shared encoder (silu) + multi-heads: prediction (y_hat), transition (f_mu), precision (pi)
- Residue memory (r), mu_prev tracking, target-driven dynamics

## ファイル

- `gnaicore.py`: コア実装 + 単一ステップ実行確認

## 使用方法

```bash
pip install jax jaxlib
python gnaicore.py
```

単一ステップでコンパイル・実行成功確認済み。シミュレーションループへの拡張が可能です。

実験は忠実に実際行って。

---

*Part of kishimoto-void cognitive geometry research family (VGE / CUBE / LDC / VoidCore).*