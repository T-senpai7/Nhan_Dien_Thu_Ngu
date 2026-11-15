"""Tien ich tao MQTT client (paho-mqtt 2.x) dung chung cho Edge, Cloud va Benchmark."""
import threading
import time
import uuid

import paho.mqtt.client as mqtt

import config


class MqttLink:
    """Boc paho client: tu ket noi lai, dem so lan reconnect, theo doi trang thai."""

    def __init__(self, role: str, subscriptions=None, on_message=None):
        self.role = role
        self.subscriptions = list(subscriptions or [])
        self._on_message = on_message
        self.connected = threading.Event()
        self.connect_count = 0
        self.disconnect_count = 0
        self.last_error = None

        client_id = f"{role}-{uuid.uuid4().hex[:8]}"
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )
        if config.MQTT_USERNAME:
            self.client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
        self.client.reconnect_delay_set(min_delay=1, max_delay=5)
        self.client.on_connect = self._handle_connect
        self.client.on_disconnect = self._handle_disconnect
        self.client.on_message = self._handle_message

    def _handle_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.last_error = str(reason_code)
            return
        self.connect_count += 1
        for topic in self.subscriptions:
            client.subscribe(topic, qos=config.MQTT_QOS)
        self.connected.set()

    def _handle_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()
        self.disconnect_count += 1
        self.last_error = str(reason_code)

    def _handle_message(self, client, userdata, msg):
        if self._on_message:
            self._on_message(msg.topic, msg.payload)

    @property
    def reconnects(self) -> int:
        return max(0, self.connect_count - 1)

    def start(self, wait_sec: float = 10.0) -> bool:
        self.client.connect_async(config.MQTT_HOST, config.MQTT_PORT, keepalive=config.MQTT_KEEPALIVE)
        self.client.loop_start()
        return self.connected.wait(wait_sec)

    def publish(self, topic: str, payload: str, qos: int = None):
        return self.client.publish(topic, payload, qos=config.MQTT_QOS if qos is None else qos)

    def stop(self):
        try:
            self.client.disconnect()
        finally:
            self.client.loop_stop()


def wait_for_broker(timeout: float = 10.0) -> bool:
    link = MqttLink("probe")
    ok = link.start(wait_sec=timeout)
    link.stop()
    time.sleep(0.1)
    return ok
