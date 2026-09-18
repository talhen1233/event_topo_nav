#!/usr/bin/env python3

import atexit
from dataclasses import dataclass
from enum import IntEnum

import serial


@dataclass
class VersionInfo:
    model: int
    hardware_version: int
    firmware_version_major: int
    firmware_version_minor: int
    serial_number: int


class BaudRateHex(IntEnum):
    BAUD_230400 = 0x00
    BAUD_460800 = 0x01
    BAUD_512000 = 0x02  # unsupported on some adapters
    BAUD_921600 = 0x03
    BAUD_1500000 = 0x04  # unsupported on some adapters


class BaudRate(IntEnum):
    BAUD_230400 = 230400
    BAUD_460800 = 460800
    BAUD_512000 = 512000  # unsupported on some adapters
    BAUD_921600 = 921600
    BAUD_1500000 = 1500000  # unsupported on some adapters


class OutputFreqHex(IntEnum):
    Freq_10Hz = 0x00
    Freq_100Hz = 0x01
    Freq_200Hz = 0x02
    Freq_500Hz = 0x03
    Freq_1000Hz = 0x04
    Freq_1800Hz = 0x05


class OutputDataFormatHex(IntEnum):
    Standard = 0x00
    Pixhawk = 0x01


class FilterHex(IntEnum):
    Off = 0x00
    On = 0x01


class CheckSumError(Exception):
    pass


class FailedToReadError(Exception):
    pass


class SelfTestFailedError(Exception):
    pass


class LidarScanningError(Exception):
    pass


PACKET_HED1 = 0xAA
PACKET_HED2 = 0x55
START_SCAN = 0x60
STOP_SCAN = 0x61
GET_DEVICE_INFO = 0x62
SELF_TEST = 0x63
SET_OUTPUT_FREQ = 0x64
SET_FILTER = 0x65
SET_SERIAL_BAUD = 0x66
SET_FORMAT_OUTPUT_DATA = 0x67
RESTORE_FACTORY_SETTINGS = 0x68
NO_DATA = 0x00


class SDM15:
    def __init__(self, port: str, baud_rate: BaudRate = BaudRate.BAUD_460800):
        self.ser = serial.Serial(port=port, baudrate=baud_rate, timeout=0.1)
        if not self.ser.is_open:
            raise Exception("Serial port is not opened")
        self.scanning = False
        self.pixhawk = False
        atexit.register(self._at_exit)

    def _at_exit(self):
        try:
            self.stop_scan()
        except Exception:
            pass
        if self.ser and self.ser.is_open:
            self.ser.close()

    @staticmethod
    def check(data: list[int]) -> int:
        return sum(data) & 0xFF

    def _reset_buffer(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def _write(self, cmd: bytes):
        self._reset_buffer()
        self.ser.write(cmd)
        self.ser.flush()

    def _read(self) -> list[int]:
        while True:
            b = self.ser.read(1)
            if not b:
                raise FailedToReadError("No data (timeout).")
            if b[0] == PACKET_HED1:
                b2 = self.ser.read(1)
                if b2 and b2[0] == PACKET_HED2:
                    break

        cmd_type = self.ser.read(1)
        data_len = self.ser.read(1)
        if not cmd_type or not data_len:
            raise FailedToReadError("Failed to read command or data_len.")
        cmd_type = cmd_type[0]
        data_len = data_len[0]

        data_and_checksum = self.ser.read(data_len + 1)
        if len(data_and_checksum) != data_len + 1:
            raise FailedToReadError(f"Expected {data_len + 1} bytes, got {len(data_and_checksum)}")

        data_segment = list(data_and_checksum[:-1])
        checksum_received = data_and_checksum[-1]
        packet_for_checksum = [PACKET_HED1, PACKET_HED2, cmd_type, data_len] + data_segment
        checksum_calculated = self.check(packet_for_checksum)
        if checksum_calculated != checksum_received:
            raise CheckSumError(f"Checksum mismatch: {checksum_received} != {checksum_calculated}")

        return [PACKET_HED1, PACKET_HED2, cmd_type, data_len] + data_segment

    def check_scanning(self):
        if self.scanning:
            raise LidarScanningError("LiDAR is scanning. Stop it before sending this command.")

    def start_scan(self):
        self._write(bytes([PACKET_HED1, PACKET_HED2, START_SCAN, NO_DATA, 0x5F]))
        self._read()
        self.scanning = True

    def stop_scan(self):
        if self.scanning:
            self._write(bytes([PACKET_HED1, PACKET_HED2, STOP_SCAN, NO_DATA, 0x60]))
            self._read()
            self.scanning = False

    def obtain_version_info(self) -> VersionInfo:
        self.check_scanning()
        self._write(bytes([PACKET_HED1, PACKET_HED2, GET_DEVICE_INFO, NO_DATA, 0x61]))
        recv = self._read()
        data_len = recv[3]
        data_segment = recv[4:4 + data_len]
        if len(data_segment) < 5:
            raise FailedToReadError("Not enough data for version info")

        serial_str = "".join(str(x) for x in data_segment[4:])
        return VersionInfo(
            model=data_segment[0],
            hardware_version=data_segment[1],
            firmware_version_major=data_segment[2],
            firmware_version_minor=data_segment[3],
            serial_number=int(serial_str) if serial_str else 0,
        )

    def lidar_self_test(self):
        self.check_scanning()
        self._write(bytes([PACKET_HED1, PACKET_HED2, SELF_TEST, NO_DATA, 0x62]))
        recv = self._read()
        data_segment = recv[4:4 + recv[3]]
        if len(data_segment) < 2:
            raise SelfTestFailedError("Missing data in self-test response.")
        if data_segment[0] != 0x01:
            raise SelfTestFailedError(f"Self-test failed, error code: {data_segment[1]}")

    def get_distance(self):
        recv = self._read()
        data_segment = recv[4:4 + recv[3]]
        if self.pixhawk:
            return 0, -1, -1
        if len(data_segment) < 4:
            raise FailedToReadError("Not enough data in distance packet")
        distance = (data_segment[1] << 8) | data_segment[0]
        return distance, data_segment[2], data_segment[3]

    def set_output_freq(self, freq: OutputFreqHex = OutputFreqHex.Freq_100Hz):
        self.check_scanning()
        cmd = [PACKET_HED1, PACKET_HED2, SET_OUTPUT_FREQ, 0x01, freq]
        cmd.append(self.check(cmd))
        self._write(bytes(cmd))
        if self._read()[4] != freq:
            raise Exception("Failed to set output frequency")
        self._reset_buffer()

    def set_filter(self, filt: FilterHex = FilterHex.On):
        self.check_scanning()
        cmd = [PACKET_HED1, PACKET_HED2, SET_FILTER, 0x01, filt]
        cmd.append(self.check(cmd))
        self._write(bytes(cmd))
        if self._read()[4] != filt:
            raise Exception("Failed to set filter")
        self._reset_buffer()

    def set_baud_rate(self, baud_rate: BaudRateHex):
        self.check_scanning()
        cmd = [PACKET_HED1, PACKET_HED2, SET_SERIAL_BAUD, 0x01, baud_rate]
        cmd.append(self.check(cmd))
        self._write(bytes(cmd))
        if self._read()[4] != baud_rate:
            raise Exception("Failed to set baud rate on LiDAR")
        self._reset_buffer()

    def set_output_data_format(self, data_format: OutputDataFormatHex):
        self.check_scanning()
        cmd = [PACKET_HED1, PACKET_HED2, SET_FORMAT_OUTPUT_DATA, 0x01, data_format]
        cmd.append(self.check(cmd))
        self._write(bytes(cmd))
        if self._read()[4] != data_format:
            raise Exception("Failed to set output data format")
        self.pixhawk = data_format == OutputDataFormatHex.Pixhawk
        self._reset_buffer()

    def restore_factory_settings(self):
        self.check_scanning()
        self._write(bytes([PACKET_HED1, PACKET_HED2, RESTORE_FACTORY_SETTINGS, 0x00, 0x67]))
        self._read()
        self._reset_buffer()
