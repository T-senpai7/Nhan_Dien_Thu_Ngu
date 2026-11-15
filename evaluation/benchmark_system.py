"""Danh gia he thong theo 3 tieu chi cua de cuong:

1. Do chinh xac (Accuracy, F1) cua Cloud AI API tren luong du lieu cam bien mo phong moi (chua tung train).
2. Do tre toan trinh: Edge -> MQTT -> Cloud AI -> MQTT -> Edge (round-trip), muc tieu < 500 ms.
3. Do on dinh MQTT: ti le goi nhan thanh cong (PDR), mat goi, thong luong o nhieu tan suat gui.

Yeu cau: broker + cloud server dang chay (python run_demo.py --no-edge, hoac chay rieng tung phan).
    python evaluation/benchmark_system.py
    python evaluation/benchmark_system.py --rates 15,50,100 --duration 20 --sequences 40
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests
from sklearn.metrics import accuracy_score, classification_report, f1_score

import config
from cloud.database import _percentiles
from edge.hand_model import SyntheticHand, simulate_sequence
from edge.sensor_simulator import SensorSimulator, build_payload
from network.mqtt_client import MqttLink


def bench_accuracy(sequences: int, seed: int) -> dict:
    print(f"\n[1] Do chinh xac Cloud AI API ({sequences} lan ra dau / cu chi, seed={seed})")
    rng = np.random.default_rng(seed)
    session = requests.Session()
    y_true, y_pred, api_ms, conf = [], [], [], []
    for label in config.WORDS:
        for _ in range(sequences):
            sim = SensorSimulator()
            frames = list(simulate_sequence(label, duration=1.5, fps=rng.uniform(15, 30), rng=rng))
            for i, (t, lm, hand) in enumerate(frames):
                feats = sim.process(lm, t, hand)
                if t >= config.MOTION_WINDOW_SEC and i % 6 == 0:
                    payload = build_payload(feats, "bench_api", i, time.time() * 1000, 0.0, label)
                    start = time.perf_counter()
                    resp = session.post(f"{config.CLOUD_URL}/api/predict", params={"mode": "words"},
                                        json=payload, timeout=5)
                    api_ms.append((time.perf_counter() - start) * 1000)
                    resp.raise_for_status()
                    body = resp.json()
                    y_true.append(label)
                    y_pred.append(body["label"])
                    conf.append(body["confidence"])
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    result = {
        "samples": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro") * 100, 2),
        "mean_confidence": round(float(np.mean(conf)) * 100, 2),
        "api_latency_ms": _percentiles(api_ms),
        "per_class": {k: {"precision": round(v["precision"] * 100, 1), "recall": round(v["recall"] * 100, 1),
                          "f1": round(v["f1-score"] * 100, 1)}
                      for k, v in report.items() if k in config.GESTURES},
    }
    print(f"    Accuracy={result['accuracy']}%  F1={result['f1_macro']}%  "
          f"HTTP API avg={result['api_latency_ms']['avg']} ms ({result['samples']} mau)")
    return result


class StreamProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.sent = {}
        self.responses = {}
        self.link = MqttLink("bench", [config.TOPIC_PREDICTION], self.on_message)

    def on_message(self, topic, raw):
        now = time.time() * 1000
        msg = json.loads(raw)
        if msg.get("type") != "prediction" or not str(msg.get("device_id", "")).startswith("bench_stream"):
            return
        with self.lock:
            key = (msg["device_id"], msg["seq_id"])
            if key in self.sent and key not in self.responses:
                self.responses[key] = {
                    "rtt": now - self.sent[key],
                    "uplink": msg["network_latency_ms"],
                    "cloud": msg["cloud_total_ms"],
                    "e2e": msg["network_latency_ms"] + msg["cloud_total_ms"],
                    "correct": msg.get("ground_truth") is None or msg["label"] == msg["ground_truth"],
                    "has_truth": msg.get("ground_truth") is not None,
                }

    def run(self, rate_hz: float, duration: float, seed: int) -> dict:
        device_id = f"bench_stream_{int(rate_hz)}hz"
        hand, sim = SyntheticHand(np.random.default_rng(seed)), SensorSimulator()
        labels = list(config.WORDS)
        start = time.time()
        hand.set_gesture(labels[0], 0.0)
        gesture_start, idx, seq = start, 0, 0
        interval = 1.0 / rate_hz
        next_t = start
        while time.time() - start < duration:
            t = time.time()
            if t - gesture_start >= 2.0:
                idx += 1
                gesture_start = t
                hand.set_gesture(labels[idx % len(labels)], t - start)
                sim.reset()
            feats = sim.process(hand.landmarks(t - start), t, hand.handedness)
            warm = t - gesture_start >= config.MOTION_WINDOW_SEC
            seq += 1
            payload = build_payload(feats, device_id, seq, t * 1000, (time.time() - t) * 1000,
                                    hand.label if warm else None)
            payload["mode"] = "words"
            with self.lock:
                self.sent[(device_id, seq)] = t * 1000
            self.link.publish(config.TOPIC_SENSOR, json.dumps(payload))
            next_t += interval
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
        time.sleep(2.0)
        elapsed = time.time() - start - 2.0
        with self.lock:
            resp = [v for (d, _), v in self.responses.items() if d == device_id]
        sent = seq
        labelled = [r for r in resp if r["has_truth"]]
        rtt = [r["rtt"] for r in resp]
        return {
            "rate_target_hz": rate_hz,
            "rate_actual_hz": round(sent / elapsed, 2),
            "duration_s": round(elapsed, 1),
            "sent": sent,
            "received": len(resp),
            "lost": sent - len(resp),
            "pdr": round(len(resp) / sent * 100, 3) if sent else 0.0,
            "e2e_ms": _percentiles([r["e2e"] for r in resp]),
            "rtt_ms": _percentiles(rtt),
            "uplink_ms": _percentiles([r["uplink"] for r in resp]),
            "cloud_ms": _percentiles([r["cloud"] for r in resp]),
            "stream_accuracy": round(np.mean([r["correct"] for r in labelled]) * 100, 2) if labelled else None,
        }


def bench_stream(rates, duration: float, seed: int) -> dict:
    print(f"\n[2+3] Do tre toan trinh & do on dinh MQTT ({config.MQTT_HOST}:{config.MQTT_PORT})")
    probe = StreamProbe()
    if not probe.link.start(wait_sec=10):
        raise SystemExit("Khong ket noi duoc MQTT broker.")
    results = []
    for rate in rates:
        r = probe.run(rate, duration, seed)
        results.append(r)
        print(f"    {rate:>5.0f} Hz: gui={r['sent']} nhan={r['received']} PDR={r['pdr']}% | "
              f"E2E avg={r['e2e_ms']['avg']} p95={r['e2e_ms']['p95']} ms (<500ms={r['e2e_ms']['under_target_pct']}%) | "
              f"RTT avg={r['rtt_ms']['avg']} p95={r['rtt_ms']['p95']} ms | acc={r['stream_accuracy']}%")
    reconnects = probe.link.reconnects
    probe.link.stop()
    return {"broker": f"{config.MQTT_HOST}:{config.MQTT_PORT}", "reconnects": reconnects, "runs": results}


def write_report(result: dict) -> Path:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (config.REPORT_DIR / f"benchmark_{stamp}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Bao cao danh gia he thong - {result['time']}", "",
             f"- Broker MQTT: `{result['stream']['broker']}`",
             f"- Cloud: `{config.CLOUD_URL}` | Mo hinh: {result['model']}", ""]
    acc = result.get("accuracy")
    if acc:
        lines += ["## 1. Do chinh xac Cloud AI (du lieu mo phong moi)", "",
                  f"- So mau: **{acc['samples']}** | Accuracy: **{acc['accuracy']}%** | F1 macro: **{acc['f1_macro']}%**",
                  f"- Do tin cay trung binh: {acc['mean_confidence']}% | Do tre HTTP API: "
                  f"avg {acc['api_latency_ms']['avg']} ms, p95 {acc['api_latency_ms']['p95']} ms", "",
                  "| Cu chi | Precision | Recall | F1 |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['precision']} | {v['recall']} | {v['f1']} |" for k, v in acc["per_class"].items()]
        lines.append("")
    lines += ["## 2. Do tre toan trinh", "",
              "- **E2E** = Edge -> MQTT -> Cloud AI -> san sang day xuong Dashboard (uplink + xu ly Cloud).",
              "- **RTT** = Edge -> MQTT -> Cloud AI -> MQTT -> Edge (di qua mang 2 lan, can tren bi quan).",
              "- Chua gom thoi gian chup + MediaPipe tren camera that (~20-40 ms); Dashboard do truc tiep gia tri nay.", "",
              "| Tan suat | Uplink avg | Cloud avg | E2E avg | E2E p95 | E2E max | E2E < 500 ms | RTT avg | RTT p95 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in result["stream"]["runs"]:
        e, t = r["e2e_ms"], r["rtt_ms"]
        lines.append(f"| {r['rate_target_hz']:.0f} Hz | {r['uplink_ms']['avg']} | {r['cloud_ms']['avg']} | {e['avg']} | "
                     f"{e['p95']} | {e['max']} | {e['under_target_pct']}% | {t['avg']} | {t['p95']} |")
    lines += ["", "## 3. Do on dinh MQTT & kha nang xu ly luong lien tuc", "",
              "| Tan suat muc tieu | Tan suat thuc | Thoi gian (s) | Gui | Nhan | Mat | PDR | Accuracy luong |",
              "|---|---|---|---|---|---|---|---|"]
    for r in result["stream"]["runs"]:
        lines.append(f"| {r['rate_target_hz']:.0f} Hz | {r['rate_actual_hz']} Hz | {r['duration_s']} | {r['sent']} | "
                     f"{r['received']} | {r['lost']} | {r['pdr']}% | {r['stream_accuracy']}% |")
    lines += ["", f"So lan ket noi lai MQTT trong qua trinh do: {result['stream']['reconnects']}", ""]
    path = config.REPORT_DIR / f"benchmark_{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Benchmark he thong IoT thu ngu")
    parser.add_argument("--rates", default="15,50,100", help="Cac tan suat gui MQTT can do (Hz)")
    parser.add_argument("--duration", type=float, default=20, help="So giay moi muc tan suat")
    parser.add_argument("--sequences", type=int, default=30, help="So lan ra dau moi cu chi cho bai do accuracy")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    try:
        health = requests.get(f"{config.CLOUD_URL}/api/health", timeout=5).json()
    except requests.RequestException:
        raise SystemExit(f"Cloud server chua chay tai {config.CLOUD_URL}")
    if "words" not in health.get("modes", []):
        raise SystemExit("Cloud chua co mo hinh che do tu ngu (can cho benchmark ban tay ao).")
    models = requests.get(f"{config.CLOUD_URL}/api/stats", timeout=5).json()["model"]

    result = {"time": datetime.now().isoformat(timespec="seconds"),
              "model": "; ".join(f"{config.MODES[m]['name']}: {v['model_name']} (test acc {v['accuracy'] * 100:.2f}%)"
                                 for m, v in models.items())}
    if not args.skip_accuracy:
        result["accuracy"] = bench_accuracy(args.sequences, args.seed)
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    result["stream"] = bench_stream(rates, args.duration, args.seed)
    path = write_report(result)
    print(f"\nBao cao: {path}")


if __name__ == "__main__":
    main()
