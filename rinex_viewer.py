"""Desktop viewer for static GNSS RINEX observation files (RINEX 2/3/4)."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import georinex as gr
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import dates as mdates


SYSTEMS = {
    "G": "GPS",
    "R": "GLONASS",
    "E": "Galileo",
    "C": "BeiDou",
    "J": "QZSS",
    "I": "NavIC",
    "S": "SBAS",
}

VIEWS = ["Observation values", "Satellite visibility", "Satellites per epoch"]


def y_axis_label(obs_code):
    """Return the usual unit of a RINEX observation code."""
    return {
        "C": "Pseudorange (m)",
        "P": "Pseudorange (m)",
        "L": "Carrier phase (cycles)",
        "D": "Doppler (Hz)",
        "S": "Signal strength (dB-Hz, when provided)",
    }.get(obs_code[:1].upper(), "Observation value")


class RinexViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("GNSS | RINEX Observation Viewer")
        self.root.geometry("1240x800")
        self.root.minsize(900, 580)

        self.dataset = None
        self.filename = None
        self.observable_names = []
        self.all_satellites = []

        self.system_var = tk.StringVar(value="All systems")
        self.satellite_var = tk.StringVar(value="All satellites")
        self.observable_var = tk.StringVar()
        self.view_var = tk.StringVar(value=VIEWS[0])
        self.info_var = tk.StringVar(value="Open a RINEX observation file to begin.")

        self.background = "#0B1220"
        self.panel = "#142033"
        self.foreground = "#E2E8F0"
        self.muted = "#94A3B8"
        self.accent = "#38BDF8"

        self._build_interface()

    def _build_interface(self):
        self.root.configure(bg=self.background)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "GNSS.TCombobox",
            fieldbackground=self.panel,
            background=self.panel,
            foreground=self.foreground,
            arrowcolor=self.accent,
            padding=7,
        )
        style.map(
            "GNSS.TCombobox",
            fieldbackground=[("readonly", self.panel)],
            foreground=[("readonly", self.foreground)],
        )
        style.configure("GNSS.TButton", font=("Segoe UI", 10, "bold"), padding=9)

        top = tk.Frame(self.root, bg=self.background)
        top.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(
            top, text="RINEX Observation Viewer", font=("Segoe UI", 19, "bold"),
            bg=self.background, fg=self.foreground
        ).pack(side="left")
        ttk.Button(top, text="Open RINEX file", style="GNSS.TButton", command=self.open_file).pack(side="right")

        controls = tk.Frame(self.root, bg=self.background)
        controls.pack(fill="x", padx=20, pady=8)
        self.system_combo = self._combobox(controls, "Constellation", self.system_var, width=19)
        self.satellite_combo = self._combobox(controls, "Satellite", self.satellite_var, width=17)
        self.observable_combo = self._combobox(controls, "Observation", self.observable_var, width=18)
        self.view_combo = self._combobox(controls, "Chart", self.view_var, width=23)
        self.view_combo["values"] = VIEWS

        self.system_combo.bind("<<ComboboxSelected>>", self._system_changed)
        for combo in (self.satellite_combo, self.observable_combo, self.view_combo):
            combo.bind("<<ComboboxSelected>>", lambda _event: self.draw())

        tk.Label(
            self.root, textvariable=self.info_var, bg=self.background,
            fg=self.muted, anchor="w", justify="left", font=("Segoe UI", 10),
        ).pack(fill="x", padx=20, pady=(3, 10))

        chart_box = tk.Frame(self.root, bg=self.panel, padx=8, pady=8)
        chart_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.figure = Figure(figsize=(12, 6.5), dpi=100, facecolor=self.panel)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, chart_box, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self._empty_chart("Choose a RINEX OBS file (.obs, .rnx, .YYo, or compressed)")

    def _combobox(self, parent, label, variable, width):
        group = tk.Frame(parent, bg=self.background)
        group.pack(side="left", padx=(0, 14))
        tk.Label(
            group, text=label, bg=self.background, fg=self.muted,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 4))
        combo = ttk.Combobox(
            group, textvariable=variable, width=width, state="readonly", style="GNSS.TCombobox"
        )
        combo.pack()
        return combo

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open static GNSS RINEX observations",
            filetypes=[("RINEX and compressed files", "*.obs *.OBS *.rnx *.RNX *.??o *.??O *.gz *.GZ *.crx *.CRX *.zip"),
                       ("All files", "*.*")],
        )
        if not path:
            return

        self.info_var.set(f"Loading {Path(path).name} ...")
        self.root.update_idletasks()
        try:
            data = gr.load(path)
            if "time" not in data.coords or "sv" not in data.coords:
                raise ValueError("This is not a RINEX observation file (OBS).")
            observable_names = [
                key for key, var in data.data_vars.items()
                if set(var.dims) == {"time", "sv"}
            ]
            if not observable_names:
                raise ValueError("No satellite observation types were found in this file.")
            satellites = [str(s) for s in data.sv.values]
            if not satellites or data.sizes.get("time", 0) == 0:
                raise ValueError("The observation file contains no epochs or satellites.")
        except Exception as exc:
            self.info_var.set("Could not open this file.")
            messagebox.showerror("RINEX read error", str(exc))
            return

        self.dataset = data
        self.filename = path
        self.observable_names = observable_names
        self.all_satellites = satellites
        prefixes = sorted({sat[0] for sat in satellites if sat})
        options = ["All systems"] + [f"{SYSTEMS.get(code, code)} ({code})" for code in prefixes]
        self.system_combo["values"] = options
        self.system_var.set("All systems")
        self.observable_combo["values"] = observable_names
        preferred = next((s for s in observable_names if s.startswith("S")), None)
        preferred = preferred or next((s for s in observable_names if s.startswith("L")), observable_names[0])
        self.observable_var.set(preferred)
        self.view_var.set(VIEWS[0])
        self._update_satellites()

        times = pd.to_datetime(data.time.values)
        self.info_var.set(
            f"{Path(path).name}   |   {data.sizes['time']:,} epochs   |   "
            f"{len(satellites)} satellites   |   {len(observable_names)} observation types\n"
            f"Start: {times[0]}   |   End: {times[-1]}   (time scale as recorded in RINEX)"
        )
        self.draw()

    def _update_satellites(self):
        choice = self.system_var.get()
        prefix = None if choice == "All systems" else choice.rsplit("(", 1)[-1].rstrip(")")
        filtered = [sv for sv in self.all_satellites if prefix is None or sv.startswith(prefix)]
        self.satellite_combo["values"] = ["All satellites"] + filtered
        self.satellite_var.set("All satellites")
        return filtered

    def _system_changed(self, _event=None):
        self._update_satellites()
        self.draw()

    def _selected_satellites(self):
        choice = self.system_var.get()
        prefix = None if choice == "All systems" else choice.rsplit("(", 1)[-1].rstrip(")")
        satellites = [sv for sv in self.all_satellites if prefix is None or sv.startswith(prefix)]
        picked = self.satellite_var.get()
        return satellites if picked == "All satellites" else [picked]

    def _style_axis(self):
        ax = self.axis
        ax.set_facecolor(self.panel)
        ax.tick_params(colors=self.foreground, labelsize=9)
        ax.xaxis.label.set_color(self.foreground)
        ax.yaxis.label.set_color(self.foreground)
        ax.title.set_color(self.foreground)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(color="#334155", alpha=0.5, linewidth=0.7)
        ax.set_axisbelow(True)

    def _empty_chart(self, message):
        self.axis.clear()
        self._style_axis()
        self.axis.text(
            0.5, 0.5, message, transform=self.axis.transAxes,
            color=self.muted, ha="center", va="center", fontsize=12,
        )
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.figure.tight_layout(pad=2)
        self.canvas.draw_idle()

    def draw(self):
        if self.dataset is None:
            return
        obs_code = self.observable_var.get()
        satellites = self._selected_satellites()
        if not satellites or obs_code not in self.dataset:
            self._empty_chart("No observations available for this selection.")
            return

        # Enforce a consistent time x satellite layout, including one-satellite cases.
        values = self.dataset[obs_code].sel(sv=satellites).transpose("time", "sv").values
        values = np.asarray(values, dtype=float)
        times = pd.to_datetime(self.dataset.time.values).to_pydatetime()
        view = self.view_var.get()
        self.axis.clear()
        self._style_axis()
        ax = self.axis

        if view == "Observation values":
            plotted = 0
            for idx, sv in enumerate(satellites):
                y = values[:, idx]
                if not np.isfinite(y).any():
                    continue
                ax.plot(times, y, linewidth=1.0, label=sv)
                plotted += 1
            ax.set_ylabel(y_axis_label(obs_code))
            ax.set_title(f"{obs_code} - Observations over time")
            if 0 < plotted <= 20:
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                          facecolor=self.panel, edgecolor="#334155", labelcolor=self.foreground,
                          ncol=1, fontsize=8)
            if plotted == 0:
                ax.text(0.5, 0.5, "No valid measurements", transform=ax.transAxes,
                        ha="center", va="center", color=self.muted)

        elif view == "Satellite visibility":
            for idx, sv in enumerate(satellites):
                mask = np.isfinite(values[:, idx])
                if np.any(mask):
                    ax.scatter(np.asarray(times)[mask], np.full(np.sum(mask), idx),
                               s=9, marker="|", color=self.accent, linewidths=1)
            ax.set_yticks(range(len(satellites)), satellites)
            ax.set_ylim(-1, len(satellites))
            ax.invert_yaxis()
            ax.set_ylabel("Satellite PRN")
            ax.set_title(f"Satellite visibility based on valid {obs_code} measurements")

        else:
            counts = np.count_nonzero(np.isfinite(values), axis=1)
            ax.step(times, counts, where="mid", linewidth=1.5, color=self.accent)
            ax.set_ylabel("Number of satellites")
            ax.set_ylim(bottom=0, top=max(int(counts.max()) + 1, 1))
            ax.set_title(f"Satellites with valid {obs_code} per epoch")

        ax.set_xlabel("Observation time (RINEX epoch)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        self.figure.tight_layout(pad=2.5)
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    RinexViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
