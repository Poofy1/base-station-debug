"""Live pose source: attaches to a running SteamVR as a background app.

Never launches SteamVR. While vrserver is down, read() returns None and the
source retries quietly; when SteamVR quits we let go and wait for it to return.
"""
from __future__ import annotations

import math
import time

from ..model import DeviceClass, DeviceInfo, Frame, LogEvent, Sample

RETRY_S = 3.0
PROPS_EVERY_S = 1.0

_CLASS = {1: DeviceClass.HMD, 2: DeviceClass.CONTROLLER, 3: DeviceClass.TRACKER, 4: DeviceClass.BASE}
_TRACKER_ROLES = ("waist", "left_foot", "right_foot", "chest", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_knee", "right_knee", "camera", "keyboard", "handed")


def mat_to_quat(m) -> tuple[float, float, float, float]:
    """Rotation part of an HmdMatrix34_t -> (w, x, y, z)."""
    m00, m01, m02 = m[0][0], m[0][1], m[0][2]
    m10, m11, m12 = m[1][0], m[1][1], m[1][2]
    m20, m21, m22 = m[2][0], m[2][1], m[2][2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


class OpenVRSource:
    def __init__(self, rate: float = 200.0) -> None:
        import openvr  # deferred so sim/replay work without the runtime installed
        self.vr = openvr
        self.period = 1.0 / rate
        self.sys = None
        self._poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self._event = openvr.VREvent_t()
        self._devices: dict[str, DeviceInfo] = {}
        self._serial_at: dict[int, str] = {}
        self._next_tick = 0.0
        self._next_props = 0.0
        self._next_retry = 0.0
        self.last_error = ""

    # ---- connection ----------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.sys is not None

    def _connect(self) -> bool:
        now = time.monotonic()
        if now < self._next_retry:
            return False
        self._next_retry = now + RETRY_S
        try:
            self.sys = self.vr.init(self.vr.VRApplication_Background)
        except self.vr.OpenVRError as e:
            self.last_error = type(e).__name__
            self.sys = None
            return False
        self.last_error = ""
        self._next_props = 0.0
        self._next_tick = time.perf_counter()
        return True

    def _disconnect(self) -> None:
        if self.sys is not None:
            try:
                self.vr.shutdown()
            except Exception:
                pass
        self.sys = None
        self._serial_at.clear()
        for d in self._devices.values():
            d.connected = False

    def _pump_events(self) -> bool:
        """Returns False if SteamVR is quitting."""
        while self.sys.pollNextEvent(self._event):
            if self._event.eventType == self.vr.VREvent_Quit:
                try:
                    self.sys.acknowledgeQuit_Exiting()
                except Exception:
                    pass
                return False
            if self._event.eventType in (self.vr.VREvent_TrackedDeviceActivated,
                                         self.vr.VREvent_TrackedDeviceDeactivated,
                                         self.vr.VREvent_TrackedDeviceRoleChanged,
                                         self.vr.VREvent_TrackedDeviceUpdated):
                self._next_props = 0.0
        return True

    # ---- properties ----------------------------------------------------
    def _str(self, i: int, prop) -> str:
        try:
            return self.sys.getStringTrackedDeviceProperty(i, prop)
        except Exception:
            return ""

    def _get(self, fn, i: int, prop):
        try:
            return fn(i, prop)
        except Exception:
            return None

    def _role(self, i: int, cls: DeviceClass, controller_type: str) -> str:
        if cls is DeviceClass.HMD:
            return "head"
        if cls is DeviceClass.CONTROLLER:
            try:
                r = self.sys.getControllerRoleForTrackedDeviceIndex(i)
            except Exception:
                r = 0
            return {1: "left_hand", 2: "right_hand"}.get(r, "")
        if cls is DeviceClass.TRACKER and controller_type.startswith("vive_tracker_"):
            role = controller_type[len("vive_tracker_"):]
            return role if role in _TRACKER_ROLES else ""
        return ""

    def _refresh_devices(self) -> None:
        vr, sys_ = self.vr, self.sys
        seen: set[str] = set()
        self._serial_at.clear()
        for i in range(vr.k_unMaxTrackedDeviceCount):
            cls = _CLASS.get(sys_.getTrackedDeviceClass(i))
            if cls is None:
                continue
            serial = self._str(i, vr.Prop_SerialNumber_String)
            if not serial:
                continue
            seen.add(serial)
            self._serial_at[i] = serial
            info = self._devices.get(serial)
            if info is None:
                info = self._devices[serial] = DeviceInfo(serial=serial, device_class=cls)
            info.device_class = cls
            info.index = i
            info.connected = bool(sys_.isTrackedDeviceConnected(i))
            info.model = self._str(i, vr.Prop_ModelNumber_String) or info.model
            info.role = self._role(i, cls, self._str(i, vr.Prop_ControllerType_String))
            info.firmware = self._str(i, vr.Prop_TrackingFirmwareVersion_String) or info.firmware
            if cls is DeviceClass.BASE:
                mode = self._str(i, vr.Prop_ModeLabel_String)
                info.channel = int(mode) if mode.isdigit() else info.channel
                continue
            if self._get(sys_.getBoolTrackedDeviceProperty, i, vr.Prop_DeviceProvidesBatteryStatus_Bool):
                info.battery = self._get(sys_.getFloatTrackedDeviceProperty, i,
                                         vr.Prop_DeviceBatteryPercentage_Float)
                info.charging = self._get(sys_.getBoolTrackedDeviceProperty, i,
                                          vr.Prop_DeviceIsCharging_Bool)
            info.dongle = self._str(i, vr.Prop_ConnectedWirelessDongle_String) or info.dongle
        for serial, info in self._devices.items():
            if serial not in seen:
                info.connected = False
                info.index = -1

    # ---- Source protocol -----------------------------------------------
    def read(self) -> Frame | None:
        if self.sys is None and not self._connect():
            time.sleep(0.25)
            return None
        try:
            if not self._pump_events():
                self._disconnect()
                return None
            now = time.perf_counter()
            delay = self._next_tick - now
            if delay > 0:
                time.sleep(delay)
                self._next_tick += self.period
            else:
                self._next_tick = now + self.period   # fell behind; don't try to catch up
            mono = time.monotonic()
            if mono >= self._next_props:
                self._next_props = mono + PROPS_EVERY_S
                self._refresh_devices()
            self.sys.getDeviceToAbsoluteTrackingPose(self.vr.TrackingUniverseStanding, 0.0, self._poses)
            t = time.time()
        except self.vr.OpenVRError as e:
            self.last_error = type(e).__name__
            self._disconnect()
            return None
        samples = []
        for i, serial in self._serial_at.items():
            p = self._poses[i]
            m = p.mDeviceToAbsoluteTracking
            v, w = p.vVelocity, p.vAngularVelocity
            samples.append(Sample(
                t=t, serial=serial,
                connected=bool(p.bDeviceIsConnected), valid=bool(p.bPoseIsValid),
                result=int(p.eTrackingResult),
                pos=(m[0][3], m[1][3], m[2][3]), quat=mat_to_quat(m),
                vel=(v[0], v[1], v[2]), angvel=(w[0], w[1], w[2])))
        return Frame(t, samples)

    def devices(self) -> dict[str, DeviceInfo]:
        return self._devices

    def drain_log(self) -> list[LogEvent]:
        return []

    def close(self) -> None:
        self._disconnect()
