"""RoboTwin evaluation hooks for the LingBot VLA v2 WebSocket server."""

import atexit
from pathlib import Path
import socket

import numpy as np
import websocket

from .msgpack_numpy import packb, unpackb


PROTOCOL = "lingbot-vla-v2/robotwin-v1"
CAMERAS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


class LingbotClient:
    """One synchronous client per simulation; the model stays in its own env."""

    def __init__(self, host="127.0.0.1", port=8006, connect_timeout=30,
                 request_timeout=300):
        uri = f"ws://{host}:{int(port)}"
        # Connect directly to the configured policy host, even when the shell
        # contains HTTP_PROXY (common on this training machine).
        connection = socket.create_connection((host, int(port)), timeout=float(connect_timeout))
        try:
            self._ws = websocket.create_connection(
                uri, timeout=float(connect_timeout), socket=connection)
        except BaseException:
            connection.close()
            raise
        self._ws.settimeout(float(request_timeout))
        try:
            self.metadata = self._receive()
            if self.metadata.get("protocol") != PROTOCOL:
                raise RuntimeError(f"Unexpected inference server at {uri}: {self.metadata}. "
                                   "Start policy/linbot-vla-v2/serve.py first.")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)
        print(f"LingBot VLA v2 connected: {uri}; checkpoint={self.metadata.get('checkpoint')}")

    def _receive(self):
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot inference server error:\n{response}")
        result = unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"Expected a dictionary from inference server, got {type(result)}")
        return result

    def infer(self, observation):
        self._ws.send_binary(packb(observation))
        return self._receive()

    def reset(self):
        self.infer({"reset": True, "robo_name": "robotwin"})

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None


def encode_obs(observation, instruction):
    """Keep training camera order and raw absolute qpos; server normalizes."""
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 14-dimensional aloha-agilex qpos, got {state.shape}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("RoboTwin task instruction must be a non-empty string")
    result = {"observation.state": state, "task": instruction}
    for camera, key in CAMERAS.items():
        rgb = np.asarray(observation["observation"][camera]["rgb"])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"{camera} must contain HWC uint8 RGB; got {rgb.shape}, {rgb.dtype}")
        result[key] = np.ascontiguousarray(rgb)
    return result


def get_model(usr_args):
    for key in ("left_arm_dim", "right_arm_dim"):
        if key in usr_args and int(usr_args[key]) != 6:
            raise ValueError("This checkpoint requires the 6+1+6+1 aloha-agilex embodiment")
    client = LingbotClient(**{key: usr_args[key] for key in
                            ("host", "port", "connect_timeout", "request_timeout")
                            if key in usr_args})
    try:
        for key in ("checkpoint", "precision", "use_length", "compile", "seed"):
            expected = usr_args.get("expected_" + key)
            if expected is None:
                continue
            actual = client.metadata.get(key)
            if key == "checkpoint" and actual is not None:
                actual, expected = str(Path(actual).resolve()), str(Path(expected).resolve())
            if actual != expected:
                raise ValueError(f"Server {key} mismatch: expected {expected}, got {actual}")
    except BaseException:
        client.close()
        raise
    return client


def eval(TASK_ENV, model, observation):
    response = model.infer(encode_obs(observation, TASK_ENV.get_instruction()))
    actions = np.asarray(response["action"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if (actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 14
            or not np.isfinite(actions).all()):
        raise ValueError(f"Expected finite (chunk, 14) actions, got {actions.shape}")
    for action in actions:
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        # [left 6 joints, left gripper, right 6 joints, right gripper]
        TASK_ENV.take_action(action, action_type="qpos")


def reset_model(model):
    model.reset()
