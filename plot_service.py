# plot_service.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, weibull_min, probplot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =========================
# Konfiguration & Optionen
# =========================

@dataclass(frozen=True)
class PlotOptions:
    plot_hist: bool = True
    plot_gauss: bool = True
    plot_weibull: bool = True
    plot_box: bool = True
    plot_qq: bool = True
    plot_grouped_bar: bool = True


@dataclass
class PlotConfig:
    folder_ids: List[str]
    filenames: List[str]
    versions: List[str] = field(default_factory=lambda: ["python", "CC"])
    bins: int = 256
    colors: Dict[str, str] = field(default_factory=dict)
    outdir: str = "Plots"

    def labels(self) -> List[str]:
        # Reihenfolge der vier Kurven, z.B. ["python_ref","python_ref_ai","CC_ref","CC_ref_ai"]
        return [f"{v}_{f}" for v in self.versions for f in self.filenames]
    

    def ensure_colors(self) -> Dict[str, str]:
        if self.colors:
            return dict(self.colors)
        default_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        lbls = self.labels()
        return {lbls[i]: default_palette[i % len(default_palette)] for i in range(len(lbls))}

# =========================
# Plot-Service
# =========================

class PlotService:
    # ------- Public API -------------------------------------

    @classmethod
    def overlay_plots(cls, config: PlotConfig, options: PlotOptions) -> None:
        """
        Erzeugt pro filename alle gewünschten Overlay-Plots (PNG).
        Für jeden filename werden die in config.versions angegebenen
        Versionen (z.B. 'python' und 'CC') gemeinsam geplottet.
        """
        colors = config.ensure_colors()
        os.makedirs(config.outdir, exist_ok=True)

        for fid in config.folder_ids:
            data, gauss_params = cls._load_data(fid, config.filenames, config.versions)
            if not data:
                logging.warning(f"[Report] Keine Daten für {fid} gefunden.")
                continue

            data_min, data_max, x = cls._get_common_range(data)

            if options.plot_hist:
                cls._plot_overlay_histogram(fid, "ALL", data, config.bins, data_min, data_max, colors, config.outdir)

            if options.plot_gauss:
                cls._plot_overlay_gauss(fid, "ALL", data, gauss_params, x, colors, config.outdir)

            if options.plot_weibull:
                cls._plot_overlay_weibull(fid, "ALL", data, x, colors, config.outdir)

            if options.plot_box:
                cls._plot_overlay_boxplot(fid, "ALL", data, colors, config.outdir)

            if options.plot_qq:
                cls._plot_overlay_qq(fid, "ALL", data, colors, config.outdir)

            if options.plot_grouped_bar:
                cls._plot_grouped_bar_means_stds(fid, "ALL", data, colors, config.outdir)

            logging.info(f"[Report] PNGs für {fid} erzeugt.")

    @classmethod
    def summary_pdf(cls, config: PlotConfig, pdf_name: str = "Plot_Vergleich.pdf") -> None:
        """
        Baut pro filename eine Seite mit allen Plots.
        """
        plot_types = [
            ("OverlayHistogramm", "Histogramm", (0, 0)),
            ("Boxplot", "Boxplot", (0, 1)),
            ("OverlayGaussFits", "Gauss-Fit", (0, 2)),
            ("OverlayWeibullFits", "Weibull-Fit", (1, 0)),
            ("QQPlot", "Q-Q-Plot", (1, 1)),
            ("GroupedBar_Mean_Std", "Mittelwert & Std Dev", (1, 2)),
        ]

        with PdfPages(pdf_name) as pdf:
            for fid in config.folder_ids:
                fig, axs = plt.subplots(2, 3, figsize=(24, 16))
                for suffix, title, (row, col) in plot_types:
                    ax = axs[row, col]
                    png = os.path.join(config.outdir, f"{fid}_ALL_{suffix}.png")
                    if os.path.exists(png):
                        img = mpimg.imread(png)
                        ax.imshow(img)
                        ax.axis("off")
                        ax.set_title(title, fontsize=22)
                    else:
                        ax.axis("off")
                        ax.set_title(f"{title}\n(nicht gefunden)", fontsize=18)
                plt.suptitle(f"{fid} – Vergleichsplots", fontsize=28)
                plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.08, hspace=0.15)
                pdf.savefig(fig)
                plt.close(fig)

        logging.info(f"[Report] Zusammenfassung gespeichert: {pdf_name}")

    # ------- Loader & Helpers --------------------------------

    @staticmethod
    def _resolve(fid: str, filename: str) -> str:
        """
        Unterstützt sowohl '<fid>/<file>' als auch 'data/<fid>/<file>'.
        """
        p1 = os.path.join(fid, filename)
        if os.path.exists(p1):
            return p1
        return os.path.join("data", fid, filename)

    @classmethod
    def _load_data(cls, fid: str, filenames: List[str], versions: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
        """
        Lädt alle Kombinationen (version x filename) und legt sie als
        separate Serien in einem einzigen Dictionary ab.
        Key = '<version>_<filename>' (z.B. 'python_ref').
        Erwartete Dateien: '<version>_<filename>_m3c2_distances.txt'
          - CC: CSV mit ';' und Spalte 'M3C2 distance' (oder erste numerische Spalte)
          - python: whitespace-getrennte Liste (np.loadtxt)
        """
        data: Dict[str, np.ndarray] = {}
        gauss_params: Dict[str, Tuple[float, float]] = {}

        for v in versions:
            for fname in filenames:
                label = f"{v}_{fname}"
                basename = f"{v}_{fname}_m3c2_distances.txt"
                path = cls._resolve(fid, basename)
                logging.info(f"[Report] Lade Daten: {path}")

                if not os.path.exists(path):
                    logging.warning(f"[Report] Datei fehlt: {path}")
                    continue

                try:
                    if v.lower() == "cc":
                        df = pd.read_csv(path, sep=";")
                        cand = [c for c in df.columns if "m3c2" in c.lower() and "distance" in c.lower()]
                        if cand:
                            col = cand[0]
                        else:
                            # Fallback: erste numerische Spalte
                            num_cols = df.select_dtypes(include=[np.number]).columns
                            if len(num_cols) == 0:
                                raise ValueError("Keine numerische Spalte gefunden.")
                            col = num_cols[0]
                        arr = df[col].astype(float).to_numpy()
                    else:
                        arr = np.loadtxt(path)
                except Exception as e:
                    logging.error(f"[Report] Laden fehlgeschlagen ({path}): {e}")
                    continue

                arr = arr[~np.isnan(arr)]
                if arr.size:
                    data[label] = arr
                    mu, std = norm.fit(arr)
                    gauss_params[label] = (float(mu), float(std))

        return data, gauss_params

    @staticmethod
    def _get_common_range(data: Dict[str, np.ndarray]) -> Tuple[float, float, np.ndarray]:
        all_vals = np.concatenate(list(data.values())) if data else np.array([])
        data_min, data_max = (float(np.min(all_vals)), float(np.max(all_vals))) if all_vals.size else (0.0, 1.0)
        x = np.linspace(data_min, data_max, 500)
        return data_min, data_max, x

    # ------- Einzelplots -------------------------------------

    @staticmethod
    def _plot_overlay_histogram(fid: str, fname: str, data: Dict[str, np.ndarray], bins: int,
                                data_min: float, data_max: float,
                                colors: Dict[str, str], outdir: str) -> None:
        plt.figure(figsize=(10, 6))
        for v, arr in data.items():
            plt.hist(arr, bins=bins, range=(data_min, data_max), density=True,
                     histtype="step", linewidth=2, label=v, color=colors.get(v))
        plt.title(f"Overlay Histogramm – {fid}/{fname}")
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Dichte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayHistogramm.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_gauss(fid: str, fname: str, data: Dict[str, np.ndarray],
                            gauss_params: Dict[str, Tuple[float, float]],
                            x: np.ndarray,
                            colors: Dict[str, str], outdir: str) -> None:
        plt.figure(figsize=(10, 6))
        for v in data.keys():
            if v in gauss_params:
                mu, std = gauss_params[v]
                plt.plot(x, norm.pdf(x, mu, std),
                         color=colors.get(v), linestyle="--" if v.lower() != "cc" else "-",
                         linewidth=2,
                         label=rf"{v} Gauss ($\mu$={mu:.4f}, $\sigma$={std:.4f})")
        plt.title(f"Overlay Gauss-Fits – {fid}/{fname}")
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Dichte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayGaussFits.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_weibull(fid: str, fname: str, data: Dict[str, np.ndarray],
                              x: np.ndarray,
                              colors: Dict[str, str], outdir: str) -> None:
        weibull_params: Dict[str, Tuple[float, float, float]] = {}
        for v, arr in data.items():
            try:
                a, loc, b = weibull_min.fit(arr)
                weibull_params[v] = (float(a), float(loc), float(b))
            except Exception as e:
                logging.warning(f"[Report] Weibull-Fit fehlgeschlagen ({fid}/{fname}, {v}): {e}")

        plt.figure(figsize=(10, 6))
        for v, (a, loc, b) in weibull_params.items():
            plt.plot(x, weibull_min.pdf(x, a, loc=loc, scale=b),
                     color=colors.get(v), linestyle="--" if v.lower() != "cc" else "-",
                     linewidth=2,
                     label=rf"{v} Weibull (a={a:.2f}, b={b:.4f})")
        plt.title(f"Overlay Weibull-Fits – {fid}/{fname}")
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Dichte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayWeibullFits.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_boxplot(fid: str, fname: str, data: Dict[str, np.ndarray],
                              colors: Dict[str, str], outdir: str) -> None:
        try:
            import seaborn as sns  # optional
            records = [pd.DataFrame({"Version": v, "Distanz": arr}) for v, arr in data.items()]
            if not records:
                return
            df = pd.concat(records, ignore_index=True)
            palette = {v: colors.get(v) for v in df["Version"].unique()}
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Version", y="Distanz", palette=palette, legend=False)
            plt.title(f"Boxplot – {fid}/{fname}")
            plt.xlabel("Version")
            plt.ylabel("M3C2-Distanz")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"))
            plt.close()
        except Exception:
            labels = list(data.keys())
            arrs = [data[v] for v in labels]
            plt.figure(figsize=(10, 6))
            b = plt.boxplot(arrs, labels=labels, patch_artist=True)
            for patch, v in zip(b["boxes"], labels):
                c = colors.get(v, "#aaaaaa")
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            plt.title(f"Boxplot – {fid}/{fname}")
            plt.xlabel("Version")
            plt.ylabel("M3C2-Distanz")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"))
            plt.close()

    @staticmethod
    def _plot_overlay_qq(fid: str, fname: str, data: Dict[str, np.ndarray],
                         colors: Dict[str, str], outdir: str) -> None:
        plt.figure(figsize=(10, 6))
        for v, arr in data.items():
            (osm, osr), (slope, intercept, r) = probplot(arr, dist="norm")
            plt.plot(osm, osr, marker="o", linestyle="", label=v, color=colors.get(v))
            plt.plot(osm, slope * osm + intercept, color=colors.get(v), linestyle="--", alpha=0.7)
        plt.title(f"Q-Q-Plot – {fid}/{fname}")
        plt.xlabel("Theoretische Quantile")
        plt.ylabel("Sortierte Werte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_QQPlot.png"))
        plt.close()

    @staticmethod
    def _plot_grouped_bar_means_stds(fid: str, fname: str, data: Dict[str, np.ndarray],
                                     colors: Dict[str, str], outdir: str) -> None:
        xlabels, mean, mean_no_out, std, std_no_out = [], [], [], [], []

        for v, arr in data.items():
            xlabels.append(v)
            mean.append(float(np.mean(arr)))
            std.append(float(np.std(arr)))
            rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0
            non_outlier = arr[np.abs(arr) <= (3 * rms)] if arr.size else np.array([])
            mean_no_out.append(float(np.mean(non_outlier)) if non_outlier.size else np.nan)
            std_no_out.append(float(np.std(non_outlier)) if non_outlier.size else np.nan)

        x = np.arange(len(xlabels))
        width = 0.35
        fig, ax = plt.subplots(2, 1, figsize=(max(8, len(xlabels) * 2), 7), sharex=True)

        cols = [colors.get(v, "#aaaaaa") for v in xlabels]
        ax[0].bar(x - width / 2, mean, width, label="mit Outlier", color=cols)
        ax[0].bar(x + width / 2, mean_no_out, width, label="ohne Outlier", color=cols, alpha=0.5)
        ax[0].set_ylabel("Mittelwert")
        ax[0].set_title(f"Mittelwert mit/ohne Outlier – {fid}/{fname}")
        ax[0].legend()

        ax[1].bar(x - width / 2, std, width, label="mit Outlier", color=cols)
        ax[1].bar(x + width / 2, std_no_out, width, label="ohne Outlier", color=cols, alpha=0.5)
        ax[1].set_ylabel("Standardabweichung")
        ax[1].set_title(f"Standardabweichung mit/ohne Outlier – {fid}/{fname}")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(xlabels, rotation=30, ha="right")
        ax[1].legend()

        plt.tight_layout()
        out = os.path.join(outdir, f"{fid}_{fname}_GroupedBar_Mean_Std.png")
        plt.savefig(out)
        plt.close()
        logging.info(f"[Report] Plot gespeichert: {out}")
