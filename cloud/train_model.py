"""Huan luyen mo hinh AI cho 2 che do: chu cai (du lieu camera that) va tu ngu (tong hop + camera).

    python cloud/train_model.py              # train ca hai che do
    python cloud/train_model.py --regen      # sinh lai du lieu tong hop cho nhom tu ngu
    python cloud/train_model.py --mode letters

Danh gia trung thuc: tap test gom cac PHIEN GHI chua tung dung de train (moi nhan giu lai
1 phien), du lieu tang cuong chi dung cho tap train. Sau do mo hinh duoc train lai tren
toan bo du lieu de trien khai.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataset.collect_data import generate_synthetic, load_raw, raw_to_features

MIN_FRAMES_PER_LABEL = 60


def candidate_models():
    return {
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800,
                          early_stopping=True, n_iter_no_change=20, random_state=42)),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42),
    }


def single_inference_ms(model, sample: np.ndarray, repeats: int = 200) -> float:
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    model.predict_proba(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(sample)
    return (time.perf_counter() - start) * 1000 / repeats


def build_dataset(mode: str, regen: bool, sequences: int, augment: int) -> pd.DataFrame:
    labels = config.MODES[mode]["labels"]
    frames = []
    raw = load_raw()
    if mode == "letters":
        raw = raw.copy()
        raw.loc[raw["label"].isin(config.MARKS_DYNAMIC), "label"] = config.POSE_LABEL
    raw = raw[raw["label"].isin(labels)]
    if not raw.empty:
        print(f"  Camera: {raw['session'].nunique()} phien, {len(raw)} khung -> trich xuat dac trung (+{augment} ban tang cuong)")
        frames.append(raw_to_features(raw, augment_copies=augment))
    if mode == "words":
        if regen or not config.SYNTHETIC_DATASET_PATH.exists():
            print("  Sinh du lieu tong hop cho nhom tu ngu...")
            generate_synthetic(sequences).to_csv(config.SYNTHETIC_DATASET_PATH, index=False)
        syn = pd.read_csv(config.SYNTHETIC_DATASET_PATH)
        if list(syn.columns[:len(config.FEATURE_COLUMNS)]) != config.FEATURE_COLUMNS:
            print("  Du lieu tong hop cu khong khop dac trung moi -> sinh lai")
            syn = generate_synthetic(sequences)
            syn.to_csv(config.SYNTHETIC_DATASET_PATH, index=False)
        frames.append(syn)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_holdout(df: pd.DataFrame):
    """Moi (nhan, nguon) co >= 2 phien: giu ~20% phien (it nhat 1) lam test. Neu khong: chia theo doan 1.5s."""
    sessions = df.groupby(["label", "source"])["session"].unique()
    if all(len(s) >= 2 for s in sessions):
        test_sessions = {x for s in sessions for x in s[-max(1, round(len(s) * 0.2)):]}
        test_mask = df["session"].isin(test_sessions)
        strategy = "giu lai cac phien ghi chua tung train (~20% / nhan, it nhat 1)"
    else:
        groups = df["chunk"].to_numpy()
        _, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7).split(df, groups=groups))
        test_mask = df.index.isin(df.index[test_idx])
        strategy = "chia theo doan ~1.5s (nen thu >= 2 phien / nhan de danh gia chuan hon)"
    train = df[~test_mask]
    test = df[test_mask & (df["variant"] == "orig")]
    return train, test, strategy


def train_mode(mode: str, args) -> dict:
    print(f"\n===== Che do: {config.MODES[mode]['name']} =====")
    df = build_dataset(mode, args.regen, args.sequences, args.augment)
    if df.empty:
        print("  Chua co du lieu -> bo qua. Thu mau: python dataset/collect_data.py camera")
        return None
    counts = df[df["variant"] == "orig"]["label"].value_counts()
    missing = [l for l in config.MODES[mode]["labels"] if counts.get(l, 0) < MIN_FRAMES_PER_LABEL]
    if missing:
        print(f"  CANH BAO: thieu du lieu cho {', '.join(config.GESTURES[l][0] for l in missing)} "
              f"(< {MIN_FRAMES_PER_LABEL} khung) -> cac nhan nay KHONG duoc nhan dien")
        df = df[~df["label"].isin(missing)]
    if df["label"].nunique() < 2:
        print("  Can it nhat 2 nhan co du du lieu -> bo qua")
        return None

    train, test, strategy = split_holdout(df)
    X_tr, y_tr = train[config.FEATURE_COLUMNS].to_numpy(float), train["label"].to_numpy()
    X_te, y_te = test[config.FEATURE_COLUMNS].to_numpy(float), test["label"].to_numpy()
    print(f"  Danh gia: {strategy} | train {len(X_tr)} | test {len(X_te)} | {len(config.FEATURE_COLUMNS)} dac trung")

    results = {}
    for name, model in candidate_models().items():
        start = time.time()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {"model": model, "accuracy": accuracy_score(y_te, pred),
                         "f1_macro": f1_score(y_te, pred, average="macro"),
                         "train_sec": time.time() - start,
                         "inference_ms": single_inference_ms(model, X_te[:1])}
        r = results[name]
        print(f"    {name:13s} acc={r['accuracy']:.4f} f1={r['f1_macro']:.4f} "
              f"train={r['train_sec']:.1f}s infer={r['inference_ms']:.2f}ms")

    top = max(r["accuracy"] for r in results.values())
    best_name = min((n for n, r in results.items() if r["accuracy"] >= top - 0.005),
                    key=lambda n: results[n]["inference_ms"])
    best = results[best_name]
    pred = best["model"].predict(X_te)
    labels = sorted(df["label"].unique())
    report = classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    confusions = sorted(((cm.loc[a, b], a, b) for a in labels for b in labels if a != b and cm.loc[a, b] > 0),
                        reverse=True)[:8]
    print(f"  -> Chon {best_name}: accuracy {best['accuracy'] * 100:.2f}% | F1 {best['f1_macro'] * 100:.2f}%")
    weak = [(l, report[l]["recall"]) for l in labels if report[l]["support"] and report[l]["recall"] < 0.9]
    if weak:
        print("  Nhan yeu (recall < 90%): " + ", ".join(f"{config.GESTURES[l][0]} {r * 100:.0f}%" for l, r in weak))
    if confusions:
        print("  Hay nham: " + ", ".join(f"{config.GESTURES[a][0]}->{config.GESTURES[b][0]} ({n})" for n, a, b in confusions))

    final = clone(candidate_models()[best_name])
    final.fit(df[config.FEATURE_COLUMNS].to_numpy(float), df["label"].to_numpy())
    if hasattr(final, "n_jobs"):
        final.set_params(n_jobs=1)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cm.to_csv(config.REPORT_DIR / f"confusion_matrix_{mode}.csv")
    metadata = {
        "mode": mode, "model_name": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "accuracy": best["accuracy"], "f1_macro": best["f1_macro"], "inference_ms": best["inference_ms"],
        "evaluation": strategy, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "n_total": int(len(df)), "labels": labels, "missing_labels": missing,
        "sources": df["source"].value_counts().to_dict(),
        "features": config.FEATURE_COLUMNS,
        "candidates": {n: {k: v for k, v in r.items() if k != "model"} for n, r in results.items()},
        "per_class": {l: {k: round(report[l][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                      for l in labels},
        "top_confusions": [{"true": a, "pred": b, "count": int(n)} for n, a, b in confusions],
    }
    return {"model": final, "metadata": metadata}


MIN_STROKES_PER_MARK = 5


def train_marks(args):
    """Mo hinh quy dao cho dau dong; moi phien ghi = 1 lan ve dau."""
    from cloud.trajectory import FEATURE_NAMES
    from dataset.collect_data import raw_to_strokes

    print("\n===== Dau dong (quy dao ngon tro) =====")
    raw = load_raw()
    raw = raw[raw["label"].isin(config.MARKS_DYNAMIC)]
    if raw.empty:
        print("  Chua co du lieu -> bo qua. Thu mau: python dataset/collect_data.py camera --group marks")
        return None
    df = raw_to_strokes(raw, augment_copies=args.augment_marks)
    orig = df[df["variant"] == "orig"]
    counts = orig["label"].value_counts()
    missing = [l for l in config.MARKS_DYNAMIC if counts.get(l, 0) < MIN_STROKES_PER_MARK]
    if missing:
        print(f"  CANH BAO: thieu du lieu cho {', '.join(config.GESTURES[l][0] for l in missing)} "
              f"(< {MIN_STROKES_PER_MARK} lan ve) -> khong duoc nhan dien")
        df = df[~df["label"].isin(missing)]
    if df["label"].nunique() < 2:
        print("  Can it nhat 2 dau co du du lieu -> bo qua")
        return None

    sessions = df[df["variant"] == "orig"].groupby("label")["session"].unique()
    test_sessions = {x for s in sessions for x in s[-max(1, round(len(s) * 0.2)):]}
    test_mask = df["session"].isin(test_sessions)
    train, test = df[~test_mask], df[test_mask & (df["variant"] == "orig")]
    X_tr, y_tr = train[FEATURE_NAMES].to_numpy(float), train["label"].to_numpy()
    X_te, y_te = test[FEATURE_NAMES].to_numpy(float), test["label"].to_numpy()
    print(f"  Danh gia: giu lai ~20% lan ve chua tung train | train {len(X_tr)} (da tang cuong) | test {len(X_te)}")

    results = {}
    for name, model in candidate_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {"model": model, "accuracy": accuracy_score(y_te, pred),
                         "f1_macro": f1_score(y_te, pred, average="macro"),
                         "inference_ms": single_inference_ms(model, X_te[:1])}
        print(f"    {name:13s} acc={results[name]['accuracy']:.4f} f1={results[name]['f1_macro']:.4f}")
    top = max(r["accuracy"] for r in results.values())
    best_name = min((n for n, r in results.items() if r["accuracy"] >= top - 0.005),
                    key=lambda n: results[n]["inference_ms"])
    best = results[best_name]
    labels = sorted(df["label"].unique())
    pred = best["model"].predict(X_te)
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    cm.to_csv(config.REPORT_DIR / "confusion_matrix_marks.csv")
    confusions = sorted(((cm.loc[a, b], a, b) for a in labels for b in labels if a != b and cm.loc[a, b] > 0),
                        reverse=True)[:6]
    print(f"  -> Chon {best_name}: accuracy {best['accuracy'] * 100:.2f}%")
    if confusions:
        print("  Hay nham: " + ", ".join(f"{config.GESTURES[a][0]}->{config.GESTURES[b][0]} ({n})" for n, a, b in confusions))

    final = clone(candidate_models()[best_name])
    final.fit(df[FEATURE_NAMES].to_numpy(float), df["label"].to_numpy())
    if hasattr(final, "n_jobs"):
        final.set_params(n_jobs=1)
    metadata = {
        "mode": "marks", "model_name": best_name, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "accuracy": best["accuracy"], "f1_macro": best["f1_macro"], "inference_ms": best["inference_ms"],
        "evaluation": "giu lai ~20% lan ve chua tung train", "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "labels": labels, "missing_labels": missing, "features": FEATURE_NAMES,
        "top_confusions": [{"true": a, "pred": b, "count": int(n)} for n, a, b in confusions],
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "metadata": metadata}, config.MARK_MODEL_PATH)
    print(f"  Da luu mo hinh dau dong -> {config.MARK_MODEL_PATH}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["letters", "words", "marks", "all"], default="all")
    parser.add_argument("--regen", action="store_true", help="Sinh lai du lieu tong hop (tu ngu)")
    parser.add_argument("--sequences", type=int, default=200)
    parser.add_argument("--augment", type=int, default=2, help="So ban tang cuong moi phien camera")
    parser.add_argument("--augment-marks", type=int, default=15, help="So ban tang cuong moi lan ve dau")
    args = parser.parse_args()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {"modes": {}}
    if config.MODEL_PATH.exists():
        old = joblib.load(config.MODEL_PATH)
        if "modes" in old and all(m["metadata"]["features"] == config.FEATURE_COLUMNS for m in old["modes"].values()):
            bundle = old
    modes = list(config.MODES) if args.mode == "all" else [] if args.mode == "marks" else [args.mode]
    for mode in modes:
        result = train_mode(mode, args)
        if result:
            bundle["modes"][mode] = result
    marks = train_marks(args) if args.mode in ("marks", "all") else None
    if not bundle["modes"] and not marks:
        raise SystemExit("Khong co mo hinh nao duoc train.")

    if modes and bundle["modes"]:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, config.MODEL_PATH)
        print(f"\nDa luu mo hinh ({', '.join(bundle['modes'])}) -> {config.MODEL_PATH}")
    summary = {m: {k: v for k, v in b["metadata"].items() if k != "features"} for m, b in bundle["modes"].items()}
    if marks:
        summary["marks"] = {k: v for k, v in marks.items() if k != "features"}
    (config.REPORT_DIR / "training_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
