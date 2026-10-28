"""
aisa_iot.py  -  laptop side of the AISA IoT link
=================================================

Two jobs, both used by aisa_final.py:

1. CallWatcher
   Looks at the Phone Link "Call on PC" window and tells whether the
   call is still ringing or has been answered.

   While it rings, the Mute / Keypad / Hold call buttons are greyed
   out. Once the other person answers they light up (and a timer
   starts). The watcher measures how bright those buttons are,
   compared with the always-bright red End button next to them.
   Because it measures relative to the End button, it still works if
   the window is resized or the display is scaled.

2. IoTLink
   MQTT connection to the Raspberry Pi.
      laptop -> Pi   aisa/state        {"state": "dialing", "service": "Ambulance"}
                     aisa/heartbeat    every 2 seconds
                     aisa/online       "online" / "offline" (last will)
      Pi -> laptop   aisa/button       {"event": "call", "service": 5}
                     aisa/pi_online    "online" / "offline"

Install on the laptop:
    pip install paho-mqtt pygetwindow

Handy checks (run from the laptop):
    python aisa_iot.py --image screenshot.png     classify a saved screenshot
    python aisa_iot.py --watch                    live view of the call window
    python aisa_iot.py --demo 192.168.1.50        play dialing/answered/idle
                                                  to the Pi (test the wiring
                                                  without running AISA)
"""

import json
import sys
import threading
import time

import cv2
import numpy as np


# ============================================================
# 1.  CALL WINDOW WATCHER
# ============================================================

# score = how "switched on" the Mute / Keypad / Hold call buttons look,
# compared with the End button (1.0 = as bright as End)
ANSWERED_MIN_SCORE = 0.85       # measured on a real call: about 1.2
RINGING_SCORE_RANGE = (0.25, 0.70)   # measured on a real call: about 0.5


def find_end_button(img):
    """Locate the red, round End button. Returns (cx, cy, radius) or None."""
    b, g, r = cv2.split(img)
    mask = ((r > 170) & (g < 100) & (b < 100)).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    best = None

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if w < 12 or h < 12:
            continue

        aspect = w / float(h)
        fill = area / float(w * h)

        # a filled circle with a white phone icon cut out of it
        if 0.8 <= aspect <= 1.25 and 0.6 <= fill <= 0.9:
            if best is None or area > best[0]:
                best = (area, x + w / 2.0, y + h / 2.0, (w + h) / 4.0)

    if best is None:
        return None

    return best[1], best[2], best[3]


def _contrast(gray, cx, cy, half_w, half_h):
    """how strongly text / icons stand out from the background here"""
    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, gray.shape[1]))
    y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, gray.shape[0]))

    crop = gray[y0:y1, x0:x1]

    if crop.size < 20:
        return 0.0

    med = float(np.median(crop))

    # max of both directions so light-theme (dark text) works too
    return max(float(np.percentile(crop, 97)) - med,
               med - float(np.percentile(crop, 3)))


def classify_call_window(img):
    """
    img = BGR picture of the call window (or of the whole screen).
    Returns (state, score) with state "ringing", "answered" or "unknown".
    """
    found = find_end_button(img)

    if found is None:
        return "unknown", 0.0

    cx, cy, r = found
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # the "End" label is always fully bright: use it as the reference
    reference = _contrast(gray, cx, cy + 1.6 * r, 1.5 * r, 0.45 * r)

    if reference < 25:
        return "unknown", 0.0

    labels, icons = [], []

    # Hold call, Keypad, Mute sit 3.76 radii apart, to the left of End
    for k in (1, 2, 3):
        x = cx - k * 3.76 * r
        labels.append(_contrast(gray, x, cy + 1.6 * r, 1.5 * r, 0.45 * r) / reference)
        icons.append(_contrast(gray, x, cy, 0.6 * r, 0.6 * r) / reference)

    score = 0.5 * float(np.mean(labels)) + 0.5 * float(np.mean(icons))

    if score >= ANSWERED_MIN_SCORE:
        return "answered", score

    if RINGING_SCORE_RANGE[0] <= score <= RINGING_SCORE_RANGE[1]:
        return "ringing", score

    return "unknown", score


class CallWatcher:
    """
    check() returns one of
        "no_window"  the call window is not on screen
        "ringing"    window found, buttons still greyed out
        "answered"   window found, buttons lit up
        "unknown"    window found but could not be read
    find_window / grab can be replaced (used by the tests).
    """

    def __init__(self, title="Call on PC", find_window=None, grab=None):
        self.title = title
        self.available = True
        self.last_score = 0.0
        self._warned = False

        self._find_window = find_window or self._default_find_window
        self._grab = grab or self._default_grab

        if find_window is None or grab is None:
            try:
                import pygetwindow  # noqa: F401
                import pyautogui    # noqa: F401
            except ImportError:
                self.available = False

    def _default_find_window(self):
        import pygetwindow as gw

        for win in gw.getWindowsWithTitle(self.title):
            if getattr(win, "isMinimized", False):
                continue

            if win.width > 100 and win.height > 100:
                return win

        return None

    def _default_grab(self, win, whole_screen=False):
        import pyautogui

        if whole_screen:
            shot = pyautogui.screenshot()
        else:
            shot = pyautogui.screenshot(
                region=(max(win.left, 0), max(win.top, 0), win.width, win.height)
            )

        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    def check(self):
        try:
            win = self._find_window()

            if win is None:
                return "no_window"

            state, score = classify_call_window(self._grab(win))
            self.last_score = score

            # window rectangle can be off on a scaled display:
            # fall back to looking at the whole screen once
            if state == "unknown":
                try:
                    state, score = classify_call_window(self._grab(win, True))
                    self.last_score = score
                except TypeError:
                    pass

            return state

        except Exception as error:
            if not self._warned:
                print("CallWatcher error:", error)
                self._warned = True

            return "unknown"


# ============================================================
# 2.  MQTT LINK TO THE RASPBERRY PI
# ============================================================

def _make_client(client_id):
    """works with paho-mqtt 1.x and 2.x"""
    import paho.mqtt.client as mqtt

    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


class IoTLink:

    def __init__(self, host, port=1883, topic="aisa", on_button=None, heartbeat_s=2.0):
        self.host = host
        self.port = port
        self.topic = topic
        self.on_button = on_button
        self.heartbeat_s = heartbeat_s

        self.connected = False
        self.pi_online = False

        self._client = None
        self._state = ("idle", "")
        self._stop = threading.Event()

    @property
    def status(self):
        """what the screen shows next to PI: online / offline"""
        return "online" if (self.connected and self.pi_online) else "offline"

    # ---------------- start / stop ----------------

    def start(self):
        try:
            self._client = _make_client(f"aisa-laptop-{int(time.time())}")
        except ImportError:
            print("IoT: paho-mqtt is not installed (pip install paho-mqtt) - IoT link off.")
            return False

        c = self._client

        # if the laptop dies, the broker tells the Pi
        c.will_set(f"{self.topic}/online", "offline", qos=1, retain=True)

        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message

        c.reconnect_delay_set(min_delay=1, max_delay=10)
        c.connect_async(self.host, self.port, keepalive=15)
        c.loop_start()

        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        print(f"IoT: connecting to the Pi broker at {self.host}:{self.port} ...")

        return True

    def stop(self):
        self._stop.set()

        if self._client is None:
            return

        try:
            self.publish_state("idle", "")
            self._publish("online", "offline", retain=True)
            time.sleep(0.2)
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

    # ---------------- publishing ----------------

    def _publish(self, suffix, payload, retain=False):
        if self._client is None or not self.connected:
            return

        try:
            if not isinstance(payload, str):
                payload = json.dumps(payload)

            self._client.publish(f"{self.topic}/{suffix}", payload, qos=1, retain=retain)

        except Exception as error:
            print("IoT publish error:", error)

    def publish_state(self, state, service=""):
        """state: idle | dialing | warning | answered"""
        self._state = (state, service)
        self._publish("state", {"state": state, "service": service, "ts": time.time()}, retain=True)

    def _heartbeat_loop(self):
        while not self._stop.wait(self.heartbeat_s):
            self._publish("heartbeat", {"ts": time.time(), "state": self._state[0]})

    # ---------------- callbacks (paho 1.x and 2.x) ----------------

    def _on_connect(self, client, userdata, flags, rc, *extra):
        failed = getattr(rc, "is_failure", None)

        if failed is None:
            failed = (rc != 0)

        if failed:
            print("IoT: broker refused the connection:", rc)
            return

        self.connected = True
        print("IoT: connected to the Pi broker.")

        client.subscribe(f"{self.topic}/button", qos=1)
        client.subscribe(f"{self.topic}/pi_online", qos=1)

        self._publish("online", "online", retain=True)
        self.publish_state(*self._state)

    def _on_disconnect(self, *args):
        if self.connected:
            print("IoT: lost the connection to the Pi broker (retrying).")

        self.connected = False
        self.pi_online = False

    def _on_message(self, client, userdata, msg):
        try:
            text = msg.payload.decode("utf-8", "ignore")

            if msg.topic == f"{self.topic}/pi_online":
                self.pi_online = (text.strip() == "online")

            elif msg.topic == f"{self.topic}/button":
                data = json.loads(text)

                if data.get("event") == "call" and self.on_button:
                    self.on_button(data)

        except Exception as error:
            print("IoT message error:", error)


# ============================================================
# command line helpers
# ============================================================

def _cli():
    import argparse

    p = argparse.ArgumentParser(description="AISA IoT helpers")
    p.add_argument("--image", help="classify a saved screenshot of the call window")
    p.add_argument("--watch", action="store_true", help="live view of the call window")
    p.add_argument("--demo", metavar="HOST", help="send dialing/answered/idle to the Pi at HOST")
    p.add_argument("--title", default="Call on PC")
    args = p.parse_args()

    if args.image:
        img = cv2.imread(args.image)

        if img is None:
            sys.exit("Could not read " + args.image)

        state, score = classify_call_window(img)
        print(f"{state}   (score {score:.2f})")

    elif args.watch:
        watcher = CallWatcher(args.title)

        if not watcher.available:
            sys.exit("Install:  pip install pygetwindow pyautogui")

        print("Start a call in Phone Link. Ctrl+C to stop.\n")

        while True:
            state = watcher.check()
            print(f"{time.strftime('%H:%M:%S')}  {state:10s} score {watcher.last_score:.2f}")
            time.sleep(0.5)

    elif args.demo:
        link = IoTLink(args.demo)
        link.start()

        for _ in range(50):
            if link.connected:
                break
            time.sleep(0.1)

        if not link.connected:
            sys.exit("Could not reach the broker on " + args.demo)

        for state, wait in (("dialing", 6), ("warning", 4), ("dialing", 4), ("answered", 6), ("idle", 1)):
            print("->", state)
            link.publish_state(state, "Ambulance")
            time.sleep(wait)

        link.stop()

    else:
        p.print_help()


if __name__ == "__main__":
    _cli()