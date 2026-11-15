"""Khoi chay toan bo he thong demo bang mot lenh.

    python run_demo.py                          # webcam + broker noi bo
    python run_demo.py --source synthetic       # ban tay ao (khong can camera)
    python run_demo.py --broker emqx            # dung broker cong cong broker.emqx.io
    python run_demo.py --no-edge                # chi chay broker + cloud (de chay benchmark)
"""
import argparse
import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_BROKERS = {"emqx": "broker.emqx.io", "hivemq": "broker.hivemq.com"}


def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_http(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Chay demo IoT phien dich thu ngu")
    parser.add_argument("--source", choices=["camera", "synthetic"], default="camera")
    parser.add_argument("--broker", default="local", help="local | emqx | hivemq | <hostname>")
    parser.add_argument("--no-edge", action="store_true", help="Khong chay Edge Gateway")
    parser.add_argument("--no-browser", action="store_true", help="Khong tu mo Dashboard")
    parser.add_argument("--retrain", action="store_true", help="Sinh lai dataset va train lai mo hinh")
    parser.add_argument("--show", action="store_true", help="Hien thi ban tay ao (synthetic)")
    args = parser.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if args.broker != "local":
        env["MQTT_HOST"] = PUBLIC_BROKERS.get(args.broker, args.broker)
        if "MQTT_TOPIC_PREFIX" not in env:
            node = hashlib.md5(platform.node().encode()).hexdigest()[:8]
            env["MQTT_TOPIC_PREFIX"] = f"iot/sign_language/{node}"
        # Broker cong cong gioi han ~10 msg/s moi client, vuot nguong se bi rot goi
        env.setdefault("PUBLISH_HZ", "8")
    os.environ.update(env)

    import config

    py = sys.executable
    procs = []

    def spawn(*cmd):
        p = subprocess.Popen([py, *cmd], cwd=ROOT, env=env)
        procs.append(p)
        return p

    try:
        if args.retrain or not config.MODEL_PATH.exists():
            print("== Huan luyen mo hinh AI (lan dau mat khoang 1 phut) ==")
            subprocess.run([py, "cloud/train_model.py", "--regen"], cwd=ROOT, env=env, check=True)

        if args.broker == "local":
            if port_open("127.0.0.1", config.MQTT_PORT):
                print(f"== Da co MQTT broker tai cong {config.MQTT_PORT}, dung lai broker nay ==")
            else:
                print("== Khoi dong MQTT broker noi bo ==")
                spawn("-m", "network.mini_broker", "--port", str(config.MQTT_PORT))
                time.sleep(1.0)
        else:
            print(f"== Dung broker {env['MQTT_HOST']} | topic {env['MQTT_TOPIC_PREFIX']}/# ==")

        print("== Khoi dong Cloud Server ==")
        spawn("cloud/server.py")
        if not wait_http(f"{config.CLOUD_URL}/api/health", 90):
            raise SystemExit("Cloud server khong khoi dong duoc.")
        print(f"== Dashboard: {config.CLOUD_URL} ==")
        if not args.no_browser:
            webbrowser.open(config.CLOUD_URL)

        if args.no_edge:
            print("Nhan Ctrl+C de dung.")
            while all(p.poll() is None for p in procs):
                time.sleep(1)
        else:
            print(f"== Khoi dong Edge Gateway ({args.source}) ==")
            edge_cmd = ["edge/edge_gateway.py", "--source", args.source]
            if args.show:
                edge_cmd.append("--show")
            spawn(*edge_cmd).wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Da dung he thong.")


if __name__ == "__main__":
    main()
