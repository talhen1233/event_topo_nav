#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import List

import gpiod
from smbus2 import SMBus, i2c_msg


def str_to_int_list(arg: List[str]) -> List[int]:
    return [int(x, 0) for x in arg]


def detect_device(bus_num: int, addr: int, timeout: float = 1.5) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with SMBus(bus_num) as b:
                b.write_quick(addr)
            return True
        except OSError:
            time.sleep(0.01)
    return False

def write_reg(bus: SMBus, addr: int, reg: int, val: int):
    msg = i2c_msg.write(addr, [(reg >> 8) & 0xFF, reg & 0xFF, val & 0xFF])
    bus.i2c_rdwr(msg)


def parse_args():
    p = argparse.ArgumentParser(
        description="Assign unique I²C addresses to multiple VL53L8CX sensors "
                    "that share the same bus. "
                    "All sensors must power-up on the factory address 0x29 "
                    "and have their own LPn (shutdown) GPIO line."
    )
    p.add_argument('--i2c-dev', default='/dev/i2c-0',
                   help='I²C device path (default: /dev/i2c-0)')
    p.add_argument('--gpio-chip', default='/dev/gpiochip0',
                   help='GPIO chip (libgpiod path, default: /dev/gpiochip0)')
    p.add_argument('--gpio-lines', type=lambda x: int(x, 0),
               nargs='+', required=True, help='GPIO line numbers, e.g. 3 5 7')
    p.add_argument('--new-addrs', type=lambda x: int(x, 0),
               nargs='+', required=True, help='Target 7-bit addresses, e.g. 0x2A 0x2B 0x2C')
    p.add_argument('--old-addr', type=lambda x: int(x, 0), default=0x29,
                   help='Factory address (default 0x29)')
    p.add_argument('--boot-delay', type=float, default=0.50,
                   help='Delay after LPn HIGH before accessing sensor [s] '
                        '(default 0.30)')
    p.add_argument('--reset-pulse', type=float, default=0.05,
                   help='How long to keep LPn LOW for a hard reset [s] '
                        '(default 0.005)')
    p.add_argument('--detect-timeout', type=float, default=1.5,
                   help='Timeout for I²C detect [s] (default 1.5)')
    return p.parse_args()

def main():
    if os.geteuid() != 0:
        sys.exit("Error: please run with sudo (root privileges required).")

    args = parse_args()
    
    if len(args.gpio_lines) != len(args.new_addrs):
        sys.exit("Error: --gpio-lines and --new-addrs must have the same length.")
    if len(set(args.new_addrs)) != len(args.new_addrs):
        sys.exit("Error: target addresses must be unique.")

    try:
        bus_num = int(args.i2c_dev.split('-')[-1])
    except ValueError:
        sys.exit(f"Cannot parse bus number from {args.i2c_dev}")

    already_ok = all(detect_device(bus_num, a) for a in args.new_addrs)
    old_present = detect_device(bus_num, args.old_addr)

    if already_ok and not old_present:
        print("[INFO] All sensors already at target addresses – nothing to do.")
        sys.exit(0)

    chip = gpiod.Chip(args.gpio_chip, gpiod.Chip.OPEN_BY_PATH)
    lines = [chip.get_line(l) for l in args.gpio_lines]
    for ln in lines:
        ln.request("vl53l8_multi", gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    print("[INIT] All LPn lines LOW → all sensors in reset.")

    with SMBus(bus_num) as bus:
        for idx, (ln, tgt_addr) in enumerate(zip(lines, args.new_addrs), start=1):
            gpio_num = args.gpio_lines[idx - 1]
            print(f"\n=== Sensor {idx}/{len(lines)} on GPIO {gpio_num} ===")

            ln.set_value(0)
            time.sleep(args.reset_pulse)
            ln.set_value(1)
            time.sleep(args.boot_delay)

            if not detect_device(bus_num, args.old_addr, args.detect_timeout):
                sys.exit(f"[ERR] No response on 0x{args.old_addr:02X} "
                         f"(GPIO {gpio_num}). Increase --boot-delay or "
                         "check wiring.")

            print(f"  Detected factory address 0x{args.old_addr:02X}. "
                  f"Assigning new address 0x{tgt_addr:02X}…")

            write_reg(bus, args.old_addr, 0x7FFF, 0x00)
            write_reg(bus, args.old_addr, 0x0004, tgt_addr)
            write_reg(bus, tgt_addr, 0x7FFF, 0x02)
            time.sleep(0.003)

            if detect_device(bus_num, args.old_addr, 0.05):
                sys.exit("[ERR] Factory address is still responding. "
                         "Try longer --reset-pulse.")

            print("  Address updated successfully.")

            ln.set_value(0)  # hold reset until all addresses are unique
            time.sleep(0.01)

    print("\n[FINAL] Bringing all sensors online…")
    for ln in lines:
        ln.set_value(1)
    time.sleep(args.boot_delay + 0.05)

    failures = []
    for gpio, tgt_addr in zip(args.gpio_lines, args.new_addrs):
        if detect_device(bus_num, tgt_addr, args.detect_timeout):
            print(f"GPIO {gpio:>2}: 0x{tgt_addr:02X} ✓")
        else:
            print(f"GPIO {gpio:>2}: 0x{tgt_addr:02X} ✗ NOT FOUND")
            failures.append(tgt_addr)

    if failures:
        sys.exit("[FAIL] Some sensors were not detected after re-enable.")
    print("\n[SUCCESS] Every sensor responds on its new address. Done!")

if __name__ == "__main__":
    main()
