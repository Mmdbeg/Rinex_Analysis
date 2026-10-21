"""Read RINEX 3.x/4.x observations and plot LGO-style satellite tracking charts."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Change these two settings or pass them as command-line arguments.
RINEX_FILE = "80.00O"
SATELLITE = "G03"

SYSTEM_NAMES = {
    "G": "GPS", "R": "GLONASS", "E": "Galileo", "C": "BeiDou",
    "S": "SBAS", "J": "QZSS", "I": "NavIC",
}
SYSTEM_COLORS = {
    "G": "#2878B5", "R": "#D98B30", "E": "#35A083", "C": "#955CB6",
    "S": "#7A8794", "J": "#C16576", "I": "#888E36",
}


def read_rinex(path):
    """Parse RINEX 3/4 observation records without external RINEX libraries."""
    obs_types = {}
    data = defaultdict(lambda: defaultdict(list))
    lli_events = defaultdict(lambda: defaultdict(list))
    epochs = []

    with open(path, "r", encoding="ascii", errors="replace") as file:
        first = file.readline()
        if "RINEX VERSION / TYPE" not in first or first[20:21] != "O":
            raise ValueError("File is not a RINEX observation file.")
        version = float(first[:9])
        if version < 3:
            raise ValueError("This script supports RINEX observation versions 3.x and 4.x.")

        # Read all observation codes, including header continuation lines.
        current_system = None
        for line in file:
            label = line[60:80].strip()
            if label == "END OF HEADER":
                break
            if label == "SYS / # / OBS TYPES":
                current_system = line[0].strip() or current_system
                if line[3:6].strip():
                    obs_types[current_system] = []
                obs_types.setdefault(current_system, []).extend(line[7:60].split())
        else:
            raise ValueError("RINEX header is incomplete.")

        # A look-ahead line handles both long vendor lines and continued records.
        pending = None

        def next_line():
            nonlocal pending
            if pending is not None:
                line, pending = pending, None
                return line
            return file.readline()

        while True:
            line = next_line()
            if not line:
                break
            if not line.startswith(">"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            year, month, day, hour, minute = map(int, parts[1:6])
            seconds = float(parts[6])
            flag, n_sat = int(parts[7]), int(parts[8])
            if flag not in (0, 1):
                for _ in range(n_sat):
                    next_line()
                continue
            when = datetime(year, month, day, hour, minute) + timedelta(seconds=seconds)
            epoch_index = len(epochs)
            epochs.append(when)

            for _ in range(n_sat):
                sat_line = next_line()
                if not sat_line:
                    raise ValueError("Unexpected end of satellite observation records.")
                sat = sat_line[:3].strip()
                codes = obs_types.get(sat[:1], [])
                payload = sat_line[3:].rstrip("\r\n")

                # Standard continuation rows begin with three spaces.
                while len(payload) < 16 * len(codes):
                    continuation = next_line()
                    if not continuation:
                        break
                    if continuation.startswith("   "):
                        payload += continuation[3:].rstrip("\r\n")
                    else:
                        pending = continuation
                        break

                for j, code in enumerate(codes):
                    field = payload[j * 16:(j + 1) * 16]
                    if len(field) < 14 or not field[:14].strip():
                        continue
                    try:
                        value = float(field[:14].replace("D", "E"))
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        data[sat][code].append((epoch_index, value))

                    # RINEX carrier-phase LLI bit 0 indicates loss of lock.
                    if code.startswith("L") and len(field) > 14 and field[14].isdigit():
                        if int(field[14]) & 1:
                            lli_events[sat][code].append(epoch_index)

    if not epochs or not data:
        raise ValueError("No valid RINEX observations found.")
    return epochs, data, lli_events


def most_complete_code(signals, prefix):
    """Prefer the observation code with the highest number of valid samples."""
    codes = [code for code in signals if code.startswith(prefix)]
    return max(codes, key=lambda code: len(signals[code])) if codes else None


def setup_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(axis="x", alpha=0.22)


def plot_overview(epochs, data, lli_events, folder):
    """Plot satellite tracking arcs and the number of tracked satellites."""
    sats = sorted(data.keys(), key=lambda sat: (sat[0], int(sat[1:])))
    num_epochs = len(epochs)
    time_num = mdates.date2num(epochs)
    delta_sec = np.diff(time_num) * 86400
    step_sec = float(np.median(delta_sec[delta_sec > 0])) if np.any(delta_sec > 0) else 1.0
    step_days = step_sec / 86400
    counts = np.zeros(num_epochs, dtype=int)

    height = max(7, len(sats) * 0.38 + 3.5)
    fig, (track_ax, count_ax) = plt.subplots(
        2, 1, figsize=(15, height), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True,
    )

    for row, sat in enumerate(sats):
        code = most_complete_code(data[sat], "L") or most_complete_code(data[sat], "C")
        if not code:
            continue
        indexes = np.array([item[0] for item in data[sat][code]], dtype=int)
        counts[indexes] += 1

        # Split observation arcs at missing epochs or significant time gaps.
        discontinuities = np.where(np.diff(time_num[indexes]) > 1.5 * step_days)[0] + 1
        segments = np.split(indexes, discontinuities)
        color = SYSTEM_COLORS.get(sat[0], "#666666")
        for segment in segments:
            if len(segment) == 0:
                continue
            left = time_num[segment[0]]
            width = time_num[segment[-1]] - left + step_days * 0.9
            track_ax.broken_barh([(left, width)], (row - 0.35, 0.7), facecolors=color)

        if code.startswith("L") and lli_events[sat].get(code):
            slip_indexes = sorted(set(lli_events[sat][code]))
            track_ax.scatter(time_num[slip_indexes], [row] * len(slip_indexes),
                             c="red", s=20, marker="x", zorder=5)

    track_ax.set_yticks(range(len(sats)))
    track_ax.set_yticklabels(sats)
    track_ax.set_ylim(-0.7, len(sats) - 0.3)
    track_ax.invert_yaxis()
    track_ax.set_ylabel("Satellite PRN")
    track_ax.set_title("GNSS Satellite Observation Timeline (carrier phase preferred)", pad=12)
    track_ax.grid(axis="x", alpha=0.22)
    track_ax.set_axisbelow(True)

    count_ax.step(epochs, counts, where="mid", color="#2878B5", linewidth=1.4)
    count_ax.set_ylabel("Satellites")
    count_ax.set_xlabel("Observation time (RINEX time scale)")
    count_ax.set_ylim(0, max(counts) + 2)
    setup_time_axis(count_ax)
    path = folder / "rinex_overview.png"
    fig.savefig(path, dpi=180)
    print(f"Saved: {path}")
    return fig


def plot_satellite(epochs, data, lli_events, sat, folder):
    """Plot code range, phase, Doppler, and SNR of a selected satellite."""
    if sat not in data:
        raise ValueError(f"Satellite {sat} is not in this file. Available: {', '.join(sorted(data))}")
    fields = [
        ("C", "Pseudorange (m)"),
        ("L", "Carrier phase (cycles)"),
        ("D", "Doppler (Hz)"),
        ("S", "Signal strength (dB-Hz)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for ax, (prefix, label) in zip(axes, fields):
        code = most_complete_code(data[sat], prefix)
        if code:
            points = data[sat][code]
            indices = np.array([p[0] for p in points], dtype=int)
            values = np.array([p[1] for p in points], dtype=float)
            # NaNs prevent connecting the curve across missing epochs.
            series = np.full(len(epochs), np.nan)
            series[indices] = values
            ax.plot(epochs, series, linewidth=1.0, color="#2878B5")
            if prefix == "L" and lli_events[sat].get(code):
                slips = np.array(sorted(set(lli_events[sat][code])), dtype=int)
                slips = slips[np.isfinite(series[slips])]
                ax.scatter([epochs[i] for i in slips], series[slips],
                           marker="x", color="red", label="Loss-of-lock flag", zorder=5)
                if len(slips):
                    ax.legend(loc="best")
        else:
            code = "not available"
        ax.set_title(f"{label}  |  {code}", loc="left", fontsize=11)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    setup_time_axis(axes[-1])
    axes[-1].set_xlabel("Observation time (RINEX time scale)")
    fig.suptitle(f"{sat} - Raw GNSS Observations", fontsize=15)
    path = folder / f"rinex_{sat}_signals.png"
    fig.savefig(path, dpi=180)
    print(f"Saved: {path}")
    return fig


def main():
    # Usage: python plot_rinex.py [observation_file] [satellite]
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(RINEX_FILE)
    sat = sys.argv[2].upper() if len(sys.argv) > 2 else SATELLITE
    epochs, data, lli_events = read_rinex(path)
    folder = path.resolve().parent
    print(f"Loaded: {path.name} | epochs: {len(epochs)} | satellites: {len(data)}")
    print(f"Period: {epochs[0]} to {epochs[-1]}")
    plot_overview(epochs, data, lli_events, folder)
    plot_satellite(epochs, data, lli_events, sat, folder)
    plt.show()


if __name__ == "__main__":
    main()
