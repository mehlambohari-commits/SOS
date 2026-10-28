#!/usr/bin/env python3
"""
AISA Alert Node  -  Raspberry Pi
================================

What it does
    - Red LED + buzzer while AISA is calling (blinking = ringing)
    - Red LED solid once the call is answered, buzzer stops
    - Yellow LED while you hold the button (0-3 s); at 3 s it goes out
      and a call to the General Helpline is requested from the laptop
    - Web page with the live status and recent events (http://<pi>:5000)
    - Knows when the laptop is offline (heartbeat + MQTT last will)

    state (sent by the laptop)   red LED       buzzer     yellow
    ---------------------------  ------------  ---------  ---------------
    idle                         off           off        on while button held
    dialing  (ringing)           blinking      beeping    off
    warning  (no answer yet)     blinking      off        off
    answered                     solid         off        off
    laptop offline               off           off        3 quick blinks if button pressed

Wiring (BCM numbers; physical pin in brackets)
    Button      GPIO17 (pin 11)  ->  other leg to GND (pin 9)
    Yellow LED  GPIO27 (pin 13)  ->  220 ohm  ->  LED long leg;  short leg to GND
    Red LED     GPIO22 (pin 15)  ->  220 ohm  ->  LED long leg;  short leg to GND
    Buzzer      GPIO23 (pin 16)  ->  buzzer +;                    buzzer - to GND
    (use an ACTIVE buzzer, the kind that beeps by itself when powered;
     all GND points can share the breadboard ground rail -> Pi pin 6)

One-time setup on the Pi
    sudo apt update
    sudo apt install -y mosquitto mosquitto-clients python3-paho-mqtt python3-flask python3-gpiozero
    echo -e "listener 1883\\nallow_anonymous true" | sudo tee /etc/mosquitto/conf.d/aisa.conf
    sudo systemctl enable --now mosquitto
    hostname -I        # note the Pi's IP address, put it in MQTT_HOST in aisa_final.py

Run
    python3 aisa_pi_node.py

Try it at a desk without any wiring or laptop
    python3 aisa_pi_node.py --simulate --no-mqtt
    then type:  d = dialing   w = warning   a = answered   i = idle
                b = hold the button for 3 s     q = quit

The broker accepts anyone on the network, so use your own Wi-Fi or a
phone hotspot, not a public network.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

BROKER_HOST = "localhost"       # the broker runs on this Pi
BROKER_PORT = 1883
TOPIC = "aisa"

BUTTON_PIN = 17
YELLOW_PIN = 27
RED_PIN = 22
BUZZER_PIN = 23

BUZZER_ACTIVE_HIGH = True       # False if your buzzer beeps while idle

HOLD_SECONDS = 3.0              # button hold needed to start a call
CONFIRM_TIMEOUT = 2.0           # laptop must confirm the call within this
OFFLINE_AFTER = 7.0             # no heartbeat for this long = laptop offline
BLINK_PERIOD = 0.3              # red LED / buzzer blink speed

BUTTON_SERVICE = 5              # 5 = General Helpline (same as the laptop)

DASHBOARD_PORT = 5000
LOG_FILE = "alerts_log.csv"


# ============================================================
# HARDWARE
# ============================================================

class SimHardware:
    """no GPIO: prints what the lights would do (for testing)"""

    def __init__(self, on_press=None, on_hold=None, on_release=None):
        self.red = self.yellow = self.buzzer = False
        self.history = []

    def set(self, red, yellow, buzzer):
        new = (bool(red), bool(yellow), bool(buzzer))

        if new != (self.red, self.yellow, self.buzzer):
            self.red, self.yellow, self.buzzer = new
            self.history.append(new)
            print(f"  [lights] red={'ON ' if red else 'off'} "
                  f"yellow={'ON ' if yellow else 'off'} "
                  f"buzzer={'ON ' if buzzer else 'off'}")

    def close(self):
        self.set(False, False, False)


class GpioHardware:
    """real LEDs, buzzer and button through gpiozero"""

    def __init__(self, on_press, on_hold, on_release):
        from gpiozero import LED, Buzzer, Button

        self.red_led = LED(RED_PIN)
        self.yellow_led = LED(YELLOW_PIN)
        self.buzzer_dev = Buzzer(BUZZER_PIN, active_high=BUZZER_ACTIVE_HIGH)

        self.button = Button(BUTTON_PIN, pull_up=True,
                             hold_time=HOLD_SECONDS, bounce_time=0.05)

        self.button.when_pressed = on_press
        self.button.when_held = on_hold
        self.button.when_released = on_release

    @staticmethod
    def _apply(device, on):
        if on:
            device.on()
        else:
            device.off()

    def set(self, red, yellow, buzzer):
        self._apply(self.red_led, red)
        self._apply(self.yellow_led, yellow)
        self._apply(self.buzzer_dev, buzzer)

    def close(self):
        self.set(False, False, False)


# ============================================================
# THE NODE  (all the behaviour lives here, so it can be tested)
# ============================================================

class AlertNode:

    def __init__(self, hardware_factory, publish):
        self.publish = publish

        self.state = "idle"
        self.service = ""

        self.last_heartbeat = 0.0
        self.lwt_offline = False

        self.pressing = False
        self.hold_done = False
        self.confirm_deadline = None
        self.fail_started = None

        self.alert_count = 0
        self.started = time.time()
        self.events = deque(maxlen=30)

        self.lock = threading.RLock()

        self._init_log()

        self.hw = hardware_factory(self.on_press, self.on_hold, self.on_release)

    # ---------------- helpers ----------------

    def laptop_online(self, now=None):
        now = time.time() if now is None else now
        return (not self.lwt_offline) and (now - self.last_heartbeat) < OFFLINE_AFTER

    def _init_log(self):
        try:
            if not os.path.exists(LOG_FILE):
                with open(LOG_FILE, "w", newline="") as f:
                    csv.writer(f).writerow(["time", "event", "service"])
        except Exception:
            pass

    def log(self, text, service=""):
        stamp = time.strftime("%H:%M:%S")

        with self.lock:
            self.events.appendleft({"time": stamp, "text": text})

        print(f"[{stamp}] {text}")

        try:
            with open(LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), text, service])
        except Exception:
            pass

    def _start_fail_blink(self, now):
        self.fail_started = now

    # ---------------- messages from the laptop ----------------

    def handle_state(self, data, now=None):
        now = time.time() if now is None else now

        with self.lock:
            new = str(data.get("state", "idle"))
            service = str(data.get("service", ""))

            if new != self.state:
                names = {"dialing": f"Call started: {service}",
                         "warning": "No answer detected yet",
                         "answered": f"Call answered: {service}",
                         "idle": "Back to idle"}
                self.log(names.get(new, new), service)

                if new == "dialing":
                    self.alert_count += 1

            self.state = new
            self.service = service

            # the laptop confirmed the call: stop waiting
            if new != "idle":
                self.confirm_deadline = None

    def handle_online(self, text, now=None):
        with self.lock:
            offline = (text.strip() == "offline")

            if offline != self.lwt_offline:
                self.log("AISA laptop went offline" if offline else "AISA laptop is online")

            self.lwt_offline = offline

    def handle_heartbeat(self, data=None, now=None):
        with self.lock:
            self.last_heartbeat = time.time() if now is None else now

    # ---------------- the button ----------------

    def on_press(self, now=None):
        now = time.time() if now is None else now

        with self.lock:
            if self.state != "idle":
                self.log("Button ignored: a call is in progress")
                return

            if not self.laptop_online(now):
                self.log("Button pressed but the AISA laptop is offline")
                self._start_fail_blink(now)
                return

            self.pressing = True
            self.hold_done = False

    def on_hold(self, now=None):
        """fires once, after HOLD_SECONDS of holding"""
        now = time.time() if now is None else now

        with self.lock:
            if not self.pressing:
                return

            self.hold_done = True
            self.confirm_deadline = now + CONFIRM_TIMEOUT

            self.log(f"Button held {HOLD_SECONDS:.0f} s: requesting a call")

            self.publish("button", {"event": "call", "service": BUTTON_SERVICE, "ts": now})

    def on_release(self, now=None):
        with self.lock:
            self.pressing = False
            self.hold_done = False

    # ---------------- outputs (called ~20 times a second) ----------------

    def tick(self, now=None):
        now = time.time() if now is None else now

        with self.lock:
            red = yellow = buzzer = False
            online = self.laptop_online(now)
            blink_on = int(now / BLINK_PERIOD) % 2 == 0

            # the laptop did not confirm the call in time
            if self.confirm_deadline is not None and now > self.confirm_deadline:
                self.confirm_deadline = None

                if self.state == "idle":
                    self.log("The laptop did not start the call")
                    self._start_fail_blink(now)

            # 3 quick yellow blinks = "could not start"
            if self.fail_started is not None:
                phase = int((now - self.fail_started) / 0.15)

                if phase < 6:
                    yellow = (phase % 2 == 0)
                else:
                    self.fail_started = None

            elif self.pressing and not self.hold_done and self.state == "idle" and online:
                yellow = True

            if online:
                if self.state == "dialing":
                    red = blink_on
                    buzzer = blink_on
                elif self.state == "warning":
                    red = blink_on
                elif self.state == "answered":
                    red = True

            self.hw.set(red, yellow, buzzer)

    # ---------------- for the web page ----------------

    def snapshot(self, now=None):
        now = time.time() if now is None else now

        with self.lock:
            age = now - self.last_heartbeat if self.last_heartbeat else None

            return {
                "state": self.state if self.laptop_online(now) else "offline",
                "service": self.service,
                "laptop_online": self.laptop_online(now),
                "heartbeat_age": None if age is None else round(age, 1),
                "alerts": self.alert_count,
                "uptime_min": int((now - self.started) / 60),
                "events": list(self.events),
            }


# ============================================================
# WEB DASHBOARD
# ============================================================

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AISA Alert Node</title>
<style>
 body{margin:0;font-family:Segoe UI,Roboto,sans-serif;background:#0b1020;color:#ecf1fa}
 .wrap{max-width:560px;margin:0 auto;padding:20px}
 h1{font-size:20px;margin:0 0 14px;letter-spacing:.5px}
 .card{border-radius:18px;padding:26px 20px;text-align:center;background:#1b2440;border:2px solid #323f62}
 .card b{display:block;font-size:34px;letter-spacing:1px}
 .card span{display:block;margin-top:6px;color:#94a0bc}
 .idle{border-color:#34d399}.idle b{color:#34d399}
 .dialing{border-color:#fbbf24;animation:pulse 1s infinite}.dialing b{color:#fbbf24}
 .warning{border-color:#fbbf24}.warning b{color:#fbbf24}
 .answered{border-color:#f87171}.answered b{color:#f87171}
 .offline{border-color:#5e6a86}.offline b{color:#94a0bc}
 @keyframes pulse{50%{box-shadow:0 0 22px #fbbf2455}}
 .row{display:flex;gap:10px;margin:14px 0}
 .tile{flex:1;background:#141b30;border:1px solid #323f62;border-radius:14px;padding:12px}
 .tile small{display:block;color:#5e6a86;font-size:11px;letter-spacing:.6px}
 .tile b{font-size:18px}
 table{width:100%;border-collapse:collapse;font-size:14px}
 td{padding:8px 6px;border-bottom:1px solid #232d4a}
 td:first-child{color:#5e6a86;width:70px}
 h2{font-size:13px;color:#94a0bc;letter-spacing:.8px;margin:22px 0 6px}
</style></head><body><div class="wrap">
<h1>AISA ALERT NODE</h1>
<div id="card" class="card offline"><b id="state">...</b><span id="sub"></span></div>
<div class="row">
 <div class="tile"><small>AISA LAPTOP</small><b id="laptop">-</b></div>
 <div class="tile"><small>HEARTBEAT</small><b id="hb">-</b></div>
 <div class="tile"><small>ALERTS</small><b id="count">0</b></div>
</div>
<h2>RECENT EVENTS</h2><table id="ev"></table>
</div>
<script>
const NAMES={idle:"ALL CLEAR",dialing:"CALLING",warning:"CALLING - NO ANSWER",answered:"CALL ANSWERED",offline:"LAPTOP OFFLINE"};
async function tick(){
 try{
  const s=await (await fetch('/api/status')).json();
  document.getElementById('card').className='card '+s.state;
  document.getElementById('state').textContent=NAMES[s.state]||s.state;
  document.getElementById('sub').textContent=s.service||'';
  document.getElementById('laptop').textContent=s.laptop_online?'online':'offline';
  document.getElementById('hb').textContent=s.heartbeat_age===null?'-':s.heartbeat_age+' s';
  document.getElementById('count').textContent=s.alerts;
  const t=document.getElementById('ev'); t.textContent='';
  s.events.forEach(e=>{const r=t.insertRow();r.insertCell().textContent=e.time;r.insertCell().textContent=e.text;});
 }catch(e){document.getElementById('state').textContent='NO CONNECTION';}
}
setInterval(tick,1000);tick();
</script></body></html>"""


def make_app(node):
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/status")
    def status():
        return jsonify(node.snapshot())

    return app


# ============================================================
# MQTT
# ============================================================

def make_client(client_id):
    """works with paho-mqtt 1.x and 2.x"""
    import paho.mqtt.client as mqtt

    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


def connect_mqtt(node_holder):
    client = make_client("aisa-pi-node")

    client.will_set(f"{TOPIC}/pi_online", "offline", qos=1, retain=True)

    def on_connect(c, userdata, flags, rc, *extra):
        failed = getattr(rc, "is_failure", None)

        if failed is None:
            failed = (rc != 0)

        if failed:
            print("MQTT: broker refused the connection:", rc)
            return

        print("MQTT: connected to the broker.")

        for suffix in ("state", "online", "heartbeat"):
            c.subscribe(f"{TOPIC}/{suffix}", qos=1)

        c.publish(f"{TOPIC}/pi_online", "online", qos=1, retain=True)

    def on_message(c, userdata, msg):
        node = node_holder["node"]

        if node is None:
            return

        try:
            text = msg.payload.decode("utf-8", "ignore")
            suffix = msg.topic.split("/", 1)[1]

            if suffix == "state":
                node.handle_state(json.loads(text))
            elif suffix == "online":
                node.handle_online(text)
            elif suffix == "heartbeat":
                node.handle_heartbeat()

        except Exception as error:
            print("MQTT message error:", error)

    client.on_connect = on_connect
    client.on_message = on_message

    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=15)
    client.loop_start()

    return client


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="AISA Alert Node (Raspberry Pi)")
    p.add_argument("--simulate", action="store_true", help="no GPIO: print the lights instead")
    p.add_argument("--no-mqtt", action="store_true", help="do not connect to a broker (desk test)")
    args = p.parse_args()

    use_gpio = not args.simulate

    if use_gpio:
        try:
            import gpiozero  # noqa: F401
        except ImportError:
            print("gpiozero is not installed - running in simulate mode.")
            use_gpio = False

    factory = GpioHardware if use_gpio else SimHardware

    holder = {"node": None}
    client = None

    def publish(suffix, payload):
        if client is not None:
            try:
                client.publish(f"{TOPIC}/{suffix}", json.dumps(payload), qos=1)
            except Exception as error:
                print("MQTT publish error:", error)
        else:
            print(f"  [mqtt off] would publish {suffix}: {payload}")

    node = AlertNode(factory, publish)
    holder["node"] = node

    if not args.no_mqtt:
        try:
            client = connect_mqtt(holder)
        except ImportError:
            sys.exit("paho-mqtt is missing:  sudo apt install python3-paho-mqtt")

    # web page
    try:
        app = make_app(node)
        threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=DASHBOARD_PORT,
                                   threaded=True, use_reloader=False),
            daemon=True,
        ).start()
        print(f"Dashboard: http://<this pi's address>:{DASHBOARD_PORT}")
    except ImportError:
        print("flask is missing, no web page:  sudo apt install python3-flask")

    print("AISA Alert Node running. Ctrl+C to stop.\n")

    # desk-test commands
    if args.simulate or args.no_mqtt:
        def console():
            keys = {"d": "dialing", "w": "warning", "a": "answered", "i": "idle"}

            for line in sys.stdin:
                key = line.strip().lower()

                if key in keys:
                    node.handle_heartbeat()
                    node.handle_state({"state": keys[key], "service": "Ambulance"})
                elif key == "b":
                    node.handle_heartbeat()
                    node.on_press()
                    time.sleep(HOLD_SECONDS + 0.2)
                    node.on_hold()
                    node.on_release()
                elif key == "q":
                    os._exit(0)

        threading.Thread(target=console, daemon=True).start()

        def keep_alive():
            while True:
                node.handle_heartbeat()
                time.sleep(2)

        threading.Thread(target=keep_alive, daemon=True).start()

    try:
        while True:
            node.tick()
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        node.hw.close()

        if client is not None:
            try:
                client.publish(f"{TOPIC}/pi_online", "offline", qos=1, retain=True)
                time.sleep(0.2)
                client.loop_stop()
                client.disconnect()
            except Exception:
                pass

        print("\nStopped.")


if __name__ == "__main__":
    main()