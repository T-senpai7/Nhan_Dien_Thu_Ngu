"""Cloud Database (SQLite) luu log cam bien, ket qua dich va chi so MQTT phuc vu thong ke."""
import json
import queue
import sqlite3
import threading
import time

import numpy as np

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT, seq_id INTEGER,
    edge_ts_ms REAL, cloud_receive_ms REAL,
    edge_processing_ms REAL, network_latency_ms REAL, ai_inference_ms REAL,
    cloud_total_ms REAL, e2e_latency_ms REAL,
    label TEXT, confidence REAL, stable_label TEXT,
    ground_truth TEXT, correct INTEGER,
    flex_thumb REAL, flex_index REAL, flex_middle REAL, flex_ring REAL, flex_pinky REAL,
    roll REAL, pitch REAL, yaw REAL, ax REAL, ay REAL, az REAL, speed REAL,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_dev_seq ON sensor_logs(device_id, seq_id);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON sensor_logs(cloud_receive_ms);

CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms REAL, device_id TEXT, label TEXT, text TEXT, confidence REAL, latency_ms REAL
);

CREATE TABLE IF NOT EXISTS mqtt_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms REAL, connected INTEGER, msgs_total INTEGER, msgs_lost INTEGER,
    pdr REAL, msg_rate REAL, reconnects INTEGER
);
"""

LOG_COLUMNS = [
    "device_id", "seq_id", "edge_ts_ms", "cloud_receive_ms", "edge_processing_ms",
    "network_latency_ms", "ai_inference_ms", "cloud_total_ms", "e2e_latency_ms",
    "label", "confidence", "stable_label", "ground_truth", "correct",
    "flex_thumb", "flex_index", "flex_middle", "flex_ring", "flex_pinky",
    "roll", "pitch", "yaw", "ax", "ay", "az", "speed", "payload",
]


def _percentiles(values):
    if not values:
        return {"count": 0, "avg": None, "p50": None, "p95": None, "max": None, "under_target_pct": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "avg": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(arr.max()), 2),
        "under_target_pct": round(float((arr < config.LATENCY_TARGET_MS).mean() * 100), 2),
    }


class Database:
    def __init__(self, path=config.DB_PATH):
        self.path = str(path)
        conn = self._connect()
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="db-writer")
        self._writer.start()
        self._read_lock = threading.Lock()
        self._reader = self._connect()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------ write
    def _write_loop(self):
        conn = self._connect()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                items = [self._queue.get(timeout=0.2)]
            except queue.Empty:
                continue
            while len(items) < 500:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                for sql, params in items:
                    conn.execute(sql, params)
                conn.commit()
            except sqlite3.Error as exc:
                print("[db] loi ghi:", exc)
        conn.close()

    def insert_log(self, row: dict):
        row = dict(row)
        if isinstance(row.get("payload"), dict):
            row["payload"] = json.dumps(row["payload"], ensure_ascii=False)
        placeholders = ",".join("?" for _ in LOG_COLUMNS)
        self._queue.put((f"INSERT INTO sensor_logs ({','.join(LOG_COLUMNS)}) VALUES ({placeholders})",
                         [row.get(c) for c in LOG_COLUMNS]))

    def update_e2e(self, device_id: str, seq_id: int, e2e_ms: float):
        self._queue.put(("UPDATE sensor_logs SET e2e_latency_ms=? WHERE device_id=? AND seq_id=? "
                         "AND e2e_latency_ms IS NULL", (e2e_ms, device_id, seq_id)))

    def insert_translation(self, ts_ms, device_id, label, text, confidence, latency_ms):
        self._queue.put(("INSERT INTO translations (ts_ms, device_id, label, text, confidence, latency_ms) "
                         "VALUES (?,?,?,?,?,?)", (ts_ms, device_id, label, text, confidence, latency_ms)))

    def insert_mqtt_metric(self, m: dict):
        self._queue.put(("INSERT INTO mqtt_metrics (ts_ms, connected, msgs_total, msgs_lost, pdr, msg_rate, reconnects) "
                         "VALUES (?,?,?,?,?,?,?)",
                         (m["ts_ms"], int(m["connected"]), m["msgs_total"], m["msgs_lost"],
                          m["pdr"], m["msg_rate"], m["reconnects"])))

    def close(self):
        self._stop.set()
        self._writer.join(timeout=3)
        self._reader.close()

    # ------------------------------------------------------------------ read
    def _query(self, sql, params=()):
        with self._read_lock:
            cur = self._reader.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def latency_stats(self, last_n: int = 2000) -> dict:
        rows = self._query(
            "SELECT network_latency_ms, cloud_total_ms, e2e_latency_ms, ai_inference_ms "
            "FROM sensor_logs ORDER BY id DESC LIMIT ?", (last_n,))
        return {
            "edge_to_cloud": _percentiles([r["network_latency_ms"] for r in rows if r["network_latency_ms"] is not None]),
            "cloud_pipeline": _percentiles([r["cloud_total_ms"] for r in rows if r["cloud_total_ms"] is not None]),
            "end_to_end": _percentiles([r["e2e_latency_ms"] for r in rows if r["e2e_latency_ms"] is not None]),
            "ai_inference": _percentiles([r["ai_inference_ms"] for r in rows if r["ai_inference_ms"] is not None]),
        }

    def summary(self) -> dict:
        totals = self._query("SELECT COUNT(*) AS n, MIN(cloud_receive_ms) AS first_ms, "
                             "MAX(cloud_receive_ms) AS last_ms FROM sensor_logs")[0]
        acc = self._query("SELECT COUNT(*) AS n, SUM(correct) AS ok FROM sensor_logs WHERE ground_truth IS NOT NULL")[0]
        label_counts = self._query("SELECT label, COUNT(*) AS n FROM translations GROUP BY label ORDER BY n DESC")
        prediction_counts = self._query(
            "SELECT stable_label AS label, COUNT(*) AS n FROM sensor_logs WHERE stable_label IS NOT NULL "
            "GROUP BY stable_label ORDER BY n DESC")
        per_class = self._query(
            "SELECT ground_truth AS label, COUNT(*) AS n, SUM(correct) AS ok FROM sensor_logs "
            "WHERE ground_truth IS NOT NULL GROUP BY ground_truth")
        return {
            "total_logs": totals["n"],
            "first_ms": totals["first_ms"],
            "last_ms": totals["last_ms"],
            "translations": label_counts,
            "predictions": prediction_counts,
            "stream_accuracy": {
                "samples": acc["n"],
                "accuracy": round(acc["ok"] / acc["n"] * 100, 2) if acc["n"] else None,
                "per_class": {r["label"]: round(r["ok"] / r["n"] * 100, 2) for r in per_class if r["n"]},
            },
            "latency": self.latency_stats(),
        }

    def recent_translations(self, limit: int = 30):
        return self._query("SELECT * FROM translations ORDER BY id DESC LIMIT ?", (limit,))

    def recent_logs(self, limit: int = 50):
        return self._query(
            "SELECT id, device_id, seq_id, edge_ts_ms, edge_processing_ms, network_latency_ms, ai_inference_ms, cloud_total_ms, "
            "e2e_latency_ms, label, confidence, stable_label, ground_truth FROM sensor_logs "
            "ORDER BY id DESC LIMIT ?", (limit,))

    def mqtt_history(self, limit: int = 120):
        return self._query("SELECT * FROM mqtt_metrics ORDER BY id DESC LIMIT ?", (limit,))[::-1]

    def reset(self):
        self._queue.put(("DELETE FROM sensor_logs", ()))
        self._queue.put(("DELETE FROM translations", ()))
        self._queue.put(("DELETE FROM mqtt_metrics", ()))
        while not self._queue.empty():
            time.sleep(0.05)
