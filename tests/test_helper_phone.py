import socket
import struct
import unittest
from unittest.mock import patch

from helper_phone import check_phone, read_packet


class PhoneProtocolTests(unittest.TestCase):
    def packet(self, flags, payload=b"frame", modern=False):
        sender, receiver = socket.socketpair()
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        sender.sendall(struct.pack(">QI", flags, len(payload)) + payload)
        return read_packet(receiver, modern)

    def test_video_timestamp_strips_keyframe_flag(self):
        self.assertEqual(self.packet((1 << 62) | 12345), (False, 12345, b"frame"))

    def test_config_packet_is_marked(self):
        self.assertEqual(self.packet(1 << 63), (True, 0, b"frame"))

    def test_scrcpy_four_uses_new_config_and_keyframe_bits(self):
        self.assertEqual(self.packet(1 << 62, modern=True), (True, 0, b"frame"))
        self.assertEqual(self.packet((1 << 61) | 12345, modern=True), (False, 12345, b"frame"))

    def test_truncated_stream_fails_instead_of_hanging(self):
        sender, receiver = socket.socketpair()
        self.addCleanup(receiver.close)
        sender.sendall(b"bad")
        sender.close()
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            read_packet(receiver)

    def test_oversized_packet_is_rejected(self):
        sender, receiver = socket.socketpair()
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        sender.sendall(struct.pack(">QI", 0, 20_000_000))
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            read_packet(receiver)

    def test_missing_dependency_explains_installation(self):
        with patch("helper_phone.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "android-tools scrcpy"):
                check_phone()

    def test_android_11_is_rejected_before_starting_camera(self):
        with patch("helper_phone.shutil.which", return_value="tool"), \
             patch("helper_phone.run_command", side_effect=["device", "30"]):
            with self.assertRaisesRegex(RuntimeError, "Android 12"):
                check_phone()

    def test_authorized_android_phone_returns_compatible_server(self):
        with patch("helper_phone.shutil.which", return_value="tool"), \
             patch("helper_phone.Path.is_file", return_value=True), \
             patch("helper_phone.run_command", side_effect=["device", "33", "Redmi Note 10 Pro",
                                                            "test-device", "scrcpy 3.3.4"]):
            info = check_phone()
        self.assertEqual(info["sdk"], 33)
        self.assertEqual(info["version"], "3.3.4")
