"""Phat lai cac phien ghi camera that (dataset/camera_raw.csv) qua MQTT nhu mot Edge that, de chup man hinh dashboard."""
import json, sys, time, argparse
from pathlib import Path
ROOT = Path(sys.argv[sys.argv.index('--root') + 1]) if '--root' in sys.argv else Path('.')
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import config
from edge.sensor_simulator import SensorSimulator, build_payload
from network.mqtt_client import MqttLink

ap = argparse.ArgumentParser()
ap.add_argument('--root'); ap.add_argument('--seq', required=True)
ap.add_argument('--hold', type=float, default=2.5); ap.add_argument('--hz', type=float, default=15)
ap.add_argument('--which', default='first'); ap.add_argument('--gap', type=float, default=0.0)
args = ap.parse_args()

raw = pd.read_csv(ROOT / 'dataset' / 'camera_raw.csv')
cols = [f'{a}{i}' for i in range(21) for a in 'xyz']

def frames_for(label):
    sess = raw[raw.label == label].session.unique()
    s = sess[-1] if args.which == 'last' else sess[0]
    g = raw[raw.session == s].sort_values('t')
    t = g.t.values - g.t.values[0]
    lm = g[cols].values.reshape(-1, 21, 3)
    hand = g.handedness.values
    if label.startswith('DAU_'):
        dur = t[-1]
        tt = np.concatenate([np.zeros(int(0.7 * args.hz)), np.arange(0, dur, 1 / args.hz)])
    else:
        start = max(0, (t[-1] - args.hold) / 2)
        tt = start + np.arange(0, args.hold, 1 / args.hz)
    idx = np.clip(np.searchsorted(t, tt), 0, len(t) - 1)
    return [(lm[i], hand[i]) for i in idx]

link = MqttLink('edge-replay', [])
assert link.start(10), 'MQTT not connected'
sim = SensorSimulator(); seq = 0; t0 = time.time(); last_status = 0
plan = []
for item in args.seq.split(','):
    if item.startswith('PAUSE'):
        plan.append(('PAUSE', float(item.split(':')[1]))); continue
    lab, _, mult = item.partition('*')
    fr = frames_for(lab)
    if mult: fr = fr * int(mult)
    plan.append((lab, fr))
for lab, fr in plan:
    if lab == 'PAUSE':
        sim.reset(); time.sleep(fr); continue
    print('->', lab, len(fr), flush=True)
    for lm, hand in fr:
        cap = time.time()
        feats = sim.process(lm, cap, hand)
        seq += 1
        proc = (time.time() - cap) * 1000
        p = build_payload(feats, config.DEVICE_ID, seq, cap * 1000, proc)
        link.publish(config.TOPIC_SENSOR, json.dumps(p))
        if cap - last_status > 1:
            last_status = cap
            link.publish(config.TOPIC_STATUS, json.dumps({"device_id": config.DEVICE_ID, "source": "replay", "fps": 15.0,
                          "hand_detected": True, "sent": seq, "ts_ms": cap * 1000}))
        time.sleep(max(0, 1 / args.hz - (time.time() - cap)))
    if args.gap: time.sleep(args.gap)
print('done', seq)
time.sleep(0.5); link.stop()
