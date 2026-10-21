"""Plot RINEX observations and flag candidate carrier-phase cycle slips.

Usage:
    python plot_rinex_cycleslip.py 80.00O R09
    python plot_rinex_cycleslip.py 80.00O G03 --phase L2P
    python plot_rinex_cycleslip.py 80.00O G03 --no-show

Requires: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

C = 299792458.0
SYSTEM_COLORS = {
    "G": "#2878B5", "R": "#D98B30", "E": "#35A083", "C": "#955CB6",
    "S": "#7A8794", "J": "#C16576", "I": "#888E36",
}
# Carrier frequencies in Hz, indexed by GNSS and RINEX band number.
FREQUENCIES = {
    "G": {"1": 1575.42e6, "2": 1227.60e6, "5": 1176.45e6},
    "E": {"1": 1575.42e6, "5": 1176.45e6, "6": 1278.75e6,
          "7": 1207.14e6, "8": 1191.795e6},
    "C": {"1": 1575.42e6, "2": 1561.098e6, "5": 1176.45e6,
          "6": 1268.52e6, "7": 1207.14e6, "8": 1191.795e6},
    "S": {"1": 1575.42e6, "5": 1176.45e6},
    "J": {"1": 1575.42e6, "2": 1227.60e6, "5": 1176.45e6,
          "6": 1278.75e6},
    "I": {"5": 1176.45e6, "9": 2492.028e6},
}


def read_rinex(path):
    """Read RINEX 3.x/4.x epochs, observation values, and phase LLI bits."""
    obs_types = {}
    data = defaultdict(lambda: defaultdict(list))
    lli = defaultdict(lambda: defaultdict(set))
    glonass_slots = {}
    epochs = []
    with open(path, "r", encoding="ascii", errors="replace") as file:
        first = file.readline()
        if "RINEX VERSION / TYPE" not in first or first[20:21] != "O":
            raise ValueError("Not a RINEX observation file.")
        if float(first[:9]) < 3:
            raise ValueError("This script supports RINEX 3.x and 4.x (not RINEX 2).")

        current_system = None
        for line in file:
            label = line[60:80].strip()
            if label == "END OF HEADER":
                break
            if label == "SYS / # / OBS TYPES":
                current_system = line[:1].strip() or current_system
                if line[3:6].strip():
                    obs_types[current_system] = []
                obs_types.setdefault(current_system, []).extend(line[7:60].split())
            if label == "GLONASS SLOT / FRQ #":
                tokens = line[:60].split()
                for j in range(1, len(tokens) - 1, 2):
                    if tokens[j].startswith("R"):
                        try:
                            glonass_slots[tokens[j]] = int(tokens[j + 1])
                        except ValueError:
                            pass
        else:
            raise ValueError("Missing END OF HEADER.")

        pending = None

        def next_line():
            nonlocal pending
            if pending is not None:
                result, pending = pending, None
                return result
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
            i_epoch = len(epochs)
            epochs.append(when)

            for _ in range(n_sat):
                row = next_line()
                if not row:
                    raise ValueError("Unexpected end of file.")
                sat = row[:3].strip()
                codes = obs_types.get(sat[:1], [])
                payload = row[3:].rstrip("\r\n")
                while len(payload) < len(codes) * 16:
                    continuation = next_line()
                    if not continuation:
                        break
                    if continuation.startswith("   ") and not continuation.startswith(">"):
                        payload += continuation[3:].rstrip("\r\n")
                    else:
                        pending = continuation
                        break
                for j, code in enumerate(codes):
                    field = payload[j * 16: (j + 1) * 16]
                    if not field[:14].strip():
                        continue
                    try:
                        value = float(field[:14].replace("D", "E"))
                    except ValueError:
                        continue
                    if not np.isfinite(value):
                        continue
                    data[sat][code].append((i_epoch, value))
                    # The LLI is character 15; character 16 is signal strength.
                    if code.startswith("L") and len(field) >= 15 and field[14].isdigit():
                        if int(field[14]) & 1:
                            lli[sat][code].add(i_epoch)

    if not epochs or not data:
        raise ValueError("No observations were found.")
    return epochs, data, lli, glonass_slots


def carrier_frequency(sat, code, glonass_slots):
    band = code[1]
    if sat.startswith("R") and sat in glonass_slots:
        slot = glonass_slots[sat]
        if band == "1":
            return (1602.0 + slot * 0.5625) * 1e6
        if band == "2":
            return (1246.0 + slot * 0.4375) * 1e6
    return FREQUENCIES.get(sat[0], {}).get(band)


def series_for(points, count):
    values = np.full(count, np.nan)
    if points:
        index, value = zip(*points)
        values[np.asarray(index, dtype=int)] = value
    return values


def robust_sigma(values):
    x = values[np.isfinite(values)]
    if not len(x):
        return 0.0
    mid = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - mid)))


def identify_spikes(residual, threshold):
    """Identify isolated one-epoch spikes by their adjacent opposite jumps."""
    spikes = set()
    ignored_edges = set()
    for i in range(1, len(residual) - 1):
        a, b = residual[i], residual[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if abs(a) <= threshold or abs(b) <= threshold or a * b >= 0:
            continue
        if abs(a + b) <= max(threshold, 0.08 * max(abs(a), abs(b))):
            spikes.add(i)
            ignored_edges.update([i, i + 1])
    return spikes, ignored_edges


def derive_doppler_residual(phase, doppler, epoch_seconds, typical_step):
    """Use Delta L + average Doppler * Delta t; Doppler has opposite phase rate."""
    residual = np.full(len(phase), np.nan)
    dt = np.diff(epoch_seconds)
    contiguous = (dt > 0) & (dt <= 1.6 * typical_step)
    available = (np.isfinite(phase[:-1]) & np.isfinite(phase[1:]) &
                 np.isfinite(doppler[:-1]) & np.isfinite(doppler[1:]) & contiguous)
    r = np.diff(phase) + 0.5 * (doppler[:-1] + doppler[1:]) * dt
    residual[1:][available] = r[available]
    return residual


def signal_bands(signals):
    """Choose one phase observation per frequency band, preferring longer arcs."""
    chosen = {}
    for code, points in signals.items():
        if not code.startswith("L") or len(code) < 3:
            continue
        band = code[1]
        if band not in chosen or len(points) > len(signals[chosen[band]]):
            chosen[band] = code
    return chosen


def detect_events(epochs, data, lli, glonass_slots):
    """Combine LLI, phase-Doppler and geometry-free tests.

    Red 'candidate' means a persistent step or an LLI indication, not proof.
    Orange 'outlier' means an isolated measurement spike that returns at the
    next epoch; this is not automatically considered a cycle slip.
    """
    n = len(epochs)
    seconds = np.array([(t - epochs[0]).total_seconds() for t in epochs])
    steps = np.diff(seconds)
    typical_step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1.0
    cache = {sat: {code: series_for(points, n) for code, points in signals.items()}
             for sat, signals in data.items()}
    # A record can accumulate independent detection methods.
    events = {}
    doppler_metrics = {}
    gf_metrics = defaultdict(list)
    wrap_edges = defaultdict(set)

    def record(sat, code, i, kind, method, magnitude=np.nan):
        key = (sat, code, i)
        if key not in events:
            events[key] = {"kind": kind, "methods": set(), "magnitudes": {}}
        # A confirmed LLI/step takes priority over an isolated outlier.
        if kind == "candidate" or events[key]["kind"] == "candidate":
            events[key]["kind"] = "candidate"
        elif kind == "wrap":
            events[key]["kind"] = "wrap"
        events[key]["methods"].add(method)
        if np.isfinite(magnitude):
            events[key]["magnitudes"][method] = float(abs(magnitude))

    # Explicit loss-of-lock is an independent receiver-side indicator.
    for sat, signal_lli in lli.items():
        for code, indices in signal_lli.items():
            for i in indices:
                record(sat, code, i, "candidate", "LLI")

    # A sudden phase jump not explained by Doppler is a candidate slip.
    for sat, signals in cache.items():
        for code, phase in signals.items():
            if not code.startswith("L"):
                continue
            doppler = signals.get("D" + code[1:])
            if doppler is None:
                continue
            residual = derive_doppler_residual(phase, doppler, seconds, typical_step)
            threshold = max(2.0, 8.0 * robust_sigma(residual))
            spikes, ignored = identify_spikes(residual, threshold)
            doppler_metrics[(sat, code)] = (residual, threshold, spikes)
            # Some receivers output phase modulo 8192 cycles (typically tiny
            # phase numbers between -8192 and zero). Repeated rollover is not
            # a loss of carrier lock. Do not mislabel those edges as slips.
            near_wrap = np.flatnonzero(np.isfinite(residual) &
                                        (np.abs(np.abs(residual) - 8192) < 3.0))
            finite_phase = np.abs(phase[np.isfinite(phase)])
            if len(near_wrap) >= 5 and len(finite_phase) > 30 and \
                    np.percentile(finite_phase, 95) < 10000:
                wrap_edges[sat, code].update(int(i) for i in near_wrap)
                for i in near_wrap:
                    record(sat, code, int(i), "wrap", "8192-cycle phase wrap", residual[i])
            for i in spikes:
                if i not in wrap_edges[sat, code]:
                    record(sat, code, i, "outlier", "Doppler", residual[i])
            for i in np.flatnonzero(np.isfinite(residual) & (abs(residual) > threshold)):
                if i not in ignored and i not in wrap_edges[sat, code]:
                    record(sat, code, int(i), "candidate", "Doppler", residual[i])

    # Geometry-free phase removes satellite-receiver geometric range.
    for sat, signals in cache.items():
        chosen = signal_bands(data[sat])
        bands = sorted(chosen)
        for j, band1 in enumerate(bands):
            for band2 in bands[j + 1:]:
                a, b = chosen[band1], chosen[band2]
                f1 = carrier_frequency(sat, a, glonass_slots)
                f2 = carrier_frequency(sat, b, glonass_slots)
                if not f1 or not f2 or abs(f1 - f2) < 1e5:
                    continue
                x, y = signals[a], signals[b]
                gf = x * (C / f1) - y * (C / f2)
                valid = np.isfinite(gf)
                dt = np.diff(seconds)
                contiguous = (dt > 0) & (dt <= 1.6 * typical_step)
                diffs = np.full(n, np.nan)
                delta = np.diff(gf)
                usable = valid[:-1] & valid[1:] & contiguous
                diffs[1:][usable] = delta[usable]
                if np.count_nonzero(np.isfinite(diffs)) < 15:
                    continue
                # Remove slow ionospheric evolution before thresholding.
                drift = float(np.nanmedian(diffs))
                residual = diffs - drift
                threshold = max(0.10, 8.0 * robust_sigma(residual))
                spikes, ignored = identify_spikes(residual, threshold)
                gf_metrics[sat].append((a, b, gf, residual, threshold, spikes))
                for i in np.flatnonzero(np.isfinite(residual) & (abs(residual) > threshold)):
                    i = int(i)
                    # A geometry-free pair containing a modulo-wrapped phase
                    # always has a fake discontinuity at the wrap edge.
                    if i in wrap_edges[sat, a] or i in wrap_edges[sat, b]:
                        continue
                    if i in ignored and i not in spikes:
                        continue
                    kind = "outlier" if i in spikes else "candidate"
                    # A returning edge of an isolated phase spike is not a new slip.
                    affected_spikes = [code for code in (a, b)
                                       if (sat, code) in doppler_metrics
                                       and (i in doppler_metrics[sat, code][2]
                                            or i - 1 in doppler_metrics[sat, code][2])]
                    if affected_spikes:
                        for code in affected_spikes:
                            if i in doppler_metrics[sat, code][2]:
                                record(sat, code, i, "outlier", "Geometry-free", residual[i])
                        continue

                    # A strong Doppler jump identifies which signal slipped.
                    supported = []
                    doppler_strength = []
                    for code in (a, b):
                        dop = doppler_metrics.get((sat, code))
                        if dop is None:
                            continue
                        d_res, d_thresh, _ = dop
                        if np.isfinite(d_res[i]):
                            strength = abs(d_res[i]) / d_thresh
                            doppler_strength.append((code, abs(d_res[i]), strength))
                            if strength > 1.0:
                                supported.append(code)
                    if supported:
                        targets = supported
                    elif len(doppler_strength) == 2:
                        doppler_strength.sort(key=lambda v: v[1], reverse=True)
                        top, other = doppler_strength
                        # Geometry-free can find smaller (< 2 cycle) slips.
                        # Attribute them only if one Doppler residual dominates.
                        if top[1] > 0.9 and top[1] > 2.5 * max(0.2, other[1]):
                            targets = [top[0]]
                        else:
                            continue
                    else:
                        targets = [a, b]
                    for code in targets:
                        record(sat, code, i, kind, "Geometry-free", residual[i])

    # Stable deterministic ordering simplifies comparisons and CSV export.
    return dict(sorted(events.items())), cache, doppler_metrics, gf_metrics


def setup_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(alpha=0.2)


def annotate_events(ax, epochs, values, events, sat, code):
    for kind, marker, color, label in [
        ("candidate", "x", "#D62828", "Possible cycle slip"),
        ("outlier", "o", "#EE9400", "Isolated outlier"),
    ]:
        indices = [i for (s, c, i), info in events.items()
                   if s == sat and c == code and info["kind"] == kind
                   and np.isfinite(values[i])]
        if indices:
            style = {"facecolors": "none", "edgecolors": color} if marker == "o" \
                    else {"color": color}
            ax.scatter([epochs[i] for i in indices], values[indices], s=58,
                       marker=marker, linewidths=1.8, zorder=6,
                       label=f"{label} ({len(indices)})", **style)
    wrapped = [i for (s, c, i), info in events.items()
               if s == sat and c == code and info["kind"] == "wrap"
               and np.isfinite(values[i])]
    if wrapped:
        ax.scatter([epochs[i] for i in wrapped], values[wrapped], s=18,
                   color="#757575", marker="s", zorder=5,
                   label=f"8192-cycle phase wraps ({len(wrapped)})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)


def plot_overview(epochs, data, events, output):
    sats = sorted(data, key=lambda sat: (sat[:1], int(sat[1:])))
    time_num = mdates.date2num(epochs)
    step_days = float(np.median(np.diff(time_num)[np.diff(time_num) > 0]))
    counts = np.zeros(len(epochs), dtype=int)
    fig, (ax, count_ax) = plt.subplots(
        2, 1, figsize=(15, max(7, len(sats) * 0.4 + 3)), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
    for row, sat in enumerate(sats):
        signals = data[sat]
        code = max((c for c in signals if c.startswith("L")),
                   key=lambda c: len(signals[c]), default=None)
        if code is None:
            continue
        indices = np.array([i for i, value in signals[code]], dtype=int)
        counts[indices] += 1
        splits = np.where(np.diff(indices) > 1)[0] + 1
        for chunk in np.split(indices, splits):
            if len(chunk):
                ax.broken_barh([(time_num[chunk[0]],
                                 time_num[chunk[-1]] - time_num[chunk[0]] + step_days * 0.9)],
                               (row - 0.34, 0.68), facecolors=SYSTEM_COLORS.get(sat[0], "gray"))
        for kind, color, mark in (("candidate", "#D62828", "x"),
                                  ("outlier", "#EE9400", "o"),
                                  ("wrap", "#757575", "s")):
            times = sorted({i for (s, code, i), info in events.items()
                            if s == sat and info["kind"] == kind})
            if times:
                style = {"facecolors": "none", "edgecolors": color} if mark == "o" \
                        else {"color": color}
                ax.scatter(time_num[times], [row] * len(times), marker=mark,
                           s=30, linewidths=1.4, zorder=7, **style)
    ax.set_yticks(range(len(sats)), sats)
    ax.set_ylim(-0.8, len(sats) - 0.2)
    ax.invert_yaxis()
    ax.set_ylabel("Satellite PRN")
    ax.set_title("GNSS Tracking | red x: slip candidate | orange o: outlier | gray square: phase wrap")
    ax.grid(axis="x", alpha=0.2)
    count_ax.step(epochs, counts, where="mid", color="#2878B5", linewidth=1.2)
    count_ax.set_ylabel("Satellites")
    count_ax.set_xlabel("Observation time (RINEX time scale)")
    setup_time_axis(count_ax)
    fig.savefig(output, dpi=180)
    return fig


def plot_satellite(epochs, data, cache, events, doppler_metrics, gf_metrics,
                   sat, phase_code, output):
    if sat not in data:
        raise ValueError(f"Satellite {sat} not found. Available: {', '.join(sorted(data))}")
    signals = cache[sat]
    phase_options = [code for code in signals if code.startswith("L")]
    if not phase_options:
        raise ValueError(f"No carrier-phase observations for {sat}.")
    if phase_code and phase_code not in phase_options:
        raise ValueError(f"{phase_code} not available for {sat}. Options: {', '.join(phase_options)}")
    phase_code = phase_code or max(phase_options, key=lambda c: np.isfinite(signals[c]).sum())
    suffix = phase_code[1:]
    primary_gf = [item for item in gf_metrics.get(sat, [])
                  if phase_code in item[:2]]
    primary_gf.sort(key=lambda item: np.isfinite(item[2]).sum(), reverse=True)
    gf_item = primary_gf[0] if primary_gf else None

    panels = 5 + bool(gf_item)
    fig, axes = plt.subplots(panels, 1, figsize=(15, 2.45 * panels), sharex=True,
                             constrained_layout=True)
    for ax, (code, label) in zip(axes[:4], [
        ("C" + suffix, "Pseudorange (m)"),
        (phase_code, "Carrier phase (cycles)"),
        ("D" + suffix, "Doppler (Hz)"),
        ("S" + suffix, "Signal strength (dB-Hz)"),
    ]):
        values = signals.get(code)
        if values is not None:
            ax.plot(epochs, values, linewidth=0.9, color="#2878B5")
            if code == phase_code:
                annotate_events(ax, epochs, values, events, sat, phase_code)
        ax.set_ylabel(label)
        ax.set_title(f"{label} | {code if values is not None else 'not available'}",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.2)

    residual_ax = axes[4]
    metric = doppler_metrics.get((sat, phase_code))
    if metric:
        residual, threshold, _ = metric
        residual_ax.plot(epochs, residual, linewidth=0.8, color="#626B74")
        residual_ax.axhline(threshold, color="red", ls="--", lw=0.8, alpha=0.6)
        residual_ax.axhline(-threshold, color="red", ls="--", lw=0.8, alpha=0.6)
        for kind, color, marker in (("candidate", "#D62828", "x"),
                                    ("outlier", "#EE9400", "o"),
                                    ("wrap", "#757575", "s")):
            indices = [i for (s, c, i), info in events.items()
                       if s == sat and c == phase_code and info["kind"] == kind
                       and np.isfinite(residual[i])]
            if indices:
                style = {"facecolors": "none", "edgecolors": color} if marker == "o" \
                        else {"color": color}
                residual_ax.scatter([epochs[i] for i in indices], residual[indices],
                                    marker=marker, s=45, zorder=5, **style)
        residual_ax.set_title(f"Phase-Doppler innovation | threshold +/-{threshold:.2f} cycles",
                              fontsize=10, loc="left")
    else:
        residual_ax.set_title("Phase-Doppler innovation | Doppler not available",
                              fontsize=10, loc="left")
    residual_ax.set_ylabel("Residual (cycles)")
    residual_ax.grid(alpha=0.2)

    if gf_item:
        a, b, gf, residual, threshold, spikes = gf_item
        ax = axes[5]
        ax.plot(epochs, gf, linewidth=0.9, color="#447A54")
        chosen = {i: info["kind"] for (s, code, i), info in events.items()
                  if s == sat and code in (a, b) and "Geometry-free" in info["methods"]}
        for kind, color, marker in (("candidate", "#D62828", "x"),
                                    ("outlier", "#EE9400", "o")):
            indices = sorted(i for i, status in chosen.items()
                             if status == kind and np.isfinite(gf[i]))
            if indices:
                style = {"facecolors": "none", "edgecolors": color} if marker == "o" \
                        else {"color": color}
                ax.scatter([epochs[i] for i in indices], gf[indices],
                           marker=marker, s=45, zorder=5, **style)
        ax.set_ylabel("GF phase (m)")
        ax.set_title(f"Geometry-free | {a} - {b} | jump threshold {threshold:.3f} m",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.2)

    setup_time_axis(axes[-1])
    axes[-1].set_xlabel("Observation time (RINEX time scale)")
    fig.suptitle(f"{sat} | {phase_code} | Candidate Cycle-Slip Detection", fontsize=14)
    fig.savefig(output, dpi=180)
    return fig, phase_code


def save_events(epochs, events, output):
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "satellite", "signal", "classification", "method",
                         "doppler_residual_cycles", "gf_jump_m"])
        for (sat, code, i), info in events.items():
            writer.writerow([
                epochs[i].isoformat(sep=" "), sat, code, info["kind"],
                "+".join(sorted(info["methods"])),
                f"{info['magnitudes']['Doppler']:.3f}" if "Doppler" in info["magnitudes"] else "",
                f"{info['magnitudes']['Geometry-free']:.3f}" if "Geometry-free" in info["magnitudes"] else "",
            ])


def main():
    parser = argparse.ArgumentParser(description="Plot RINEX and detect candidate cycle slips")
    parser.add_argument("rinex", type=Path, nargs="?", default=Path(__file__).with_name("80.00O"))
    parser.add_argument("satellite", nargs="?", default="G03")
    parser.add_argument("--phase", help="Carrier phase signal, e.g. L1P or L2P")
    parser.add_argument("--no-show", action="store_true", help="Only save PNG and CSV")
    args = parser.parse_args()
    sat = args.satellite.upper()
    epochs, data, lli, slots = read_rinex(args.rinex)
    events, cache, doppler_metrics, gf_metrics = detect_events(epochs, data, lli, slots)
    folder = args.rinex.resolve().parent
    overview_path = folder / "rinex_overview_cycleslips.png"
    satellite_path = folder / f"rinex_{sat}_cycleslips.png"
    csv_path = folder / "rinex_cycleslip_events.csv"
    save_events(epochs, events, csv_path)
    plot_overview(epochs, data, events, overview_path)
    fig, code = plot_satellite(epochs, data, cache, events, doppler_metrics,
                               gf_metrics, sat, args.phase, satellite_path)
    unique_candidates = {(s, i) for (s, c, i), info in events.items()
                         if info["kind"] == "candidate"}
    unique_outliers = {(s, i) for (s, c, i), info in events.items()
                       if info["kind"] == "outlier"}
    unique_wraps = {(s, i) for (s, c, i), info in events.items()
                    if info["kind"] == "wrap"}
    print(f"Read {args.rinex.name}: {len(epochs)} epochs, {len(data)} satellites")
    print(f"Cycle-slip candidates: {len(unique_candidates)} sat-epochs; "
          f"isolated outliers: {len(unique_outliers)} sat-epochs; "
          f"phase wraps: {len(unique_wraps)} sat-epochs")
    print(f"Selected: {sat} / {code}")
    for name in (overview_path, satellite_path, csv_path):
        print(f"Saved: {name}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
