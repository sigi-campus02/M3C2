# report_service.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm, weibull_min, probplot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

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
    outdir: str = "Plots"     # Zielordner für PNGs


@dataclass
class PlotConfig:
    folder: List[str]
    filenames: List[str]
    bins: int = 256
    colors: Dict[str, str] = field(default_factory=dict)

    def ensure_colors(self) -> Dict[str, str]:
        """Sorgt für eine Farbzuordnung für alle Versionen + CC (falls nicht übergeben)."""
        if self.colors:
            colors = dict(self.colors)
        else:
            # robuste Defaults – ähnlich deinem Skript
            default_palette = [
                "#1f77b4",  # blau
                "#ff7f0e",  # orange
                "#2ca02c",  # grün
                "#d62728",  # rot
                "#9467bd",  # lila
                "#8c564b",  # braun
                "#e377c2",  # pink
                "#7f7f7f",  # grau
                "#bcbd22",  # oliv
                "#17becf",  # cyan
            ]
            colors = {v: default_palette[i % len(default_palette)] for i, v in enumerate(self.folder)}
        return colors


# =========================
# Plot-Service
# =========================

class PlotService:
    # ------- Public API -------------------------------------

    @classmethod
    def overlay_plots(cls, folder_id: str, config: PlotConfig, options: PlotOptions = PlotOptions()) -> None:
        """
        Erzeugt alle gewünschten Overlay-Plots (PNG) für eine ID.
        """
        colors = config.ensure_colors()
        data, gauss_params = cls._load_data(folder_id, config.filenames)
        if not data:
            logging.warning(f"[Report] Keine Daten für ID {folder_id} gefunden.")
            return

        data_min, data_max, x = cls._get_common_range(data)

        os.makedirs(options.outdir, exist_ok=True)

        if options.plot_hist:
            cls._plot_overlay_histogram(folder_id, data, config.bins, data_min, data_max, colors, options.outdir)

        if options.plot_gauss:
            cls._plot_overlay_gauss(folder_id, data, gauss_params, x, colors, options.outdir)

        if options.plot_weibull:
            cls._plot_overlay_weibull(folder_id, data, x, colors, options.outdir)

        if options.plot_box:
            cls._plot_overlay_boxplot(folder_id, data, colors, options.outdir)

        if options.plot_qq:
            cls._plot_overlay_qq(folder_id, data, colors, options.outdir)

        if options.plot_grouped_bar:
            cls._plot_grouped_bar_means_stds(folder_id, data, colors, options.outdir)

        logging.info(f"[Report] PNGs für ID {folder_id} erzeugt.")

    @classmethod
    def summary_pdf(cls, folder_ids: List[str], pdf_name: str = "Plot_Vergleich.pdf", outdir: str = "Plots") -> None:
        """
        Baut aus vorhandenen PNGs pro ID eine Sammel-PDF.
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
            for fid in folder_ids:
                fig, axs = plt.subplots(2, 3, figsize=(24, 16))
                for suffix, title, (row, col) in plot_types:
                    ax = axs[row, col]
                    fname = os.path.join(outdir, f"{fid}_{suffix}.png")
                    if os.path.exists(fname):
                        img = mpimg.imread(fname)
                        ax.imshow(img)
                        ax.axis("off")
                        ax.set_title(title, fontsize=22)
                    else:
                        ax.axis("off")
                        ax.set_title(f"{title}\n(nicht gefunden)", fontsize=18)
                plt.suptitle(f"ID {fid} – Vergleichsplots", fontsize=28)
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
        p2 = os.path.join("data", fid, filename)
        return p2

    @classmethod
    def _load_data(cls, fid: str, filenames: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
        data: Dict[str, np.ndarray] = {}
        gauss_params: Dict[str, Tuple[float, float]] = {}

        # Python-Versionen
        for v in filenames:
            py_path = cls._resolve(fid, f"{v}_m3c2_distances.txt")
            logging.info(f"[Report] Lade Python-Daten für {v} von {py_path}")
            if os.path.exists(py_path):
                arr = np.loadtxt(py_path)
                arr = arr[~np.isnan(arr)]
                if arr.size:
                    data[v] = arr
                    mu, std = norm.fit(arr)
                    gauss_params[v] = (float(mu), float(std))

        return data, gauss_params

    @staticmethod
    def _get_common_range(data: Dict[str, np.ndarray]) -> Tuple[float, float, np.ndarray]:
        all_vals = np.concatenate([d for d in data.values()]) if data else np.array([])
        data_min, data_max = (float(np.min(all_vals)), float(np.max(all_vals))) if all_vals.size else (0.0, 1.0)
        x = np.linspace(data_min, data_max, 500)
        return data_min, data_max, x

    # ------- Einzelplots -------------------------------------

    @staticmethod
    def _plot_overlay_histogram(fid: str, data: Dict[str, np.ndarray], bins: int,
                                data_min: float, data_max: float,
                                colors: Dict[str, str], outdir: str) -> None:
        plt.figure(figsize=(10, 6))
        for v, arr in data.items():
            plt.hist(arr, bins=bins, range=(data_min, data_max), density=True,
                     histtype="step", linewidth=2, label=v, color=colors.get(v, None))
        plt.title(f"Overlay Histogramm für ID {fid}")
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Dichte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_OverlayHistogramm.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_gauss(fid: str, data: Dict[str, np.ndarray],
                            gauss_params: Dict[str, Tuple[float, float]],
                            x: np.ndarray,
                            colors: Dict[str, str], outdir: str) -> None:
        plt.figure(figsize=(10, 6))
        for v in data.keys():
            if v in gauss_params:
                mu, std = gauss_params[v]
                plt.plot(x, norm.pdf(x, mu, std),
                         color=colors.get(v, None),
                         linestyle="--" if v != "CC" else "-",
                         linewidth=2,
                         label=rf"{v} Gauss ($\mu$={mu:.4f}, $\sigma$={std:.4f})")
        plt.title(f"Overlay Gauss-Fits für ID {fid}")
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Dichte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_OverlayGaussFits.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_weibull(fid: str, data: Dict[str, np.ndarray],
                              x: np.ndarray,
                              colors: Dict[str, str], outdir: str) -> None:
        weibull_params: Dict[str, Tuple[float, float, float]] = {}
        for v, arr in data.items():
            try:
                a, loc, b = weibull_min.fit(arr)
                weibull_params[v] = (float(a), float(loc), float(b))
            except Exception as e:
                logging.warning(f"[Report] Weibull-Fit fehlgeschlagen ({fid}, {v}): {e}")

        plt.figure(figsize=(10, 6))
        for v, (a, loc, b) in weibull_params.items():
            plt.plot(x, weibull_min.pdf(x, a, loc=loc, scale=b),
                     color=colors.get(v, None),
                     linestyle="--" if v != "CC" else "-",
                     linewidth=2,
                     label=rf"{v} Weibull (a={a:.2f}, b={b:.4f})")
        plt.title(f"Overlay Weibull-Fits für ID {fid}")
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Dichte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_OverlayWeibullFits.png"))
        plt.close()

    @staticmethod
    def _plot_overlay_boxplot(fid: str, data: Dict[str, np.ndarray],
                              colors: Dict[str, str], outdir: str) -> None:
        # Versuche seaborn; fallback auf Matplotlib
        try:
            import seaborn as sns  # type: ignore
            records = []
            for v, arr in data.items():
                records.append(pd.DataFrame({"Version": v, "Distanz": arr}))
            if not records:
                return
            df = pd.concat(records, ignore_index=True)
            palette = {v: colors.get(v, None) for v in df["Version"].unique()}
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Version", y="Distanz", palette=palette, legend=False)
            plt.title(f"Vergleichender Boxplot für ID {fid}")
            plt.xlabel("Version")
            plt.ylabel("M3C2-Distanz")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_Boxplot.png"))
            plt.close()
        except Exception:
            # Matplotlib-Fallback
            labels = list(data.keys())
            arrs = [data[v] for v in labels]
            plt.figure(figsize=(10, 6))
            b = plt.boxplot(arrs, labels=labels, patch_artist=True)
            for patch, v in zip(b["boxes"], labels):
                c = colors.get(v, "#aaaaaa")
                patch.set_facecolor(c)
                patch.set_alpha(0.5)
            plt.title(f"Vergleichender Boxplot für ID {fid}")
            plt.xlabel("Version")
            plt.ylabel("M3C2-Distanz")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_Boxplot.png"))
            plt.close()

    @staticmethod
    def _plot_overlay_qq(fid: str, data: Dict[str, np.ndarray],
                         colors: Dict[str, str], outdir: str) -> None:
        plt.figure(figsize=(10, 6))
        for v, arr in data.items():
            (osm, osr), (slope, intercept, r) = probplot(arr, dist="norm")
            plt.plot(osm, osr, marker="o", linestyle="", label=v, color=colors.get(v, None))
            plt.plot(osm, slope * osm + intercept, color=colors.get(v, None), linestyle="--", alpha=0.7)
        plt.title(f"Q-Q-Plot für ID {fid}")
        plt.xlabel("Theoretische Quantile")
        plt.ylabel("Sortierte Werte")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_QQPlot.png"))
        plt.close()

    @staticmethod
    def _plot_grouped_bar_means_stds(fid: str, data: Dict[str, np.ndarray],
                                     colors: Dict[str, str], outdir: str) -> None:
        xlabels: List[str] = []
        mean: List[float] = []
        mean_no_out: List[float] = []
        std: List[float] = []
        std_no_out: List[float] = []

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

        cols = [colors.get(v, None) for v in xlabels]
        ax[0].bar(x - width / 2, mean, width, label="mit Outlier", color=cols)
        ax[0].bar(x + width / 2, mean_no_out, width, label="ohne Outlier", color=cols, alpha=0.5)
        ax[0].set_ylabel("Mittelwert")
        ax[0].set_title(f"Mittelwert mit/ohne Outlier – ID {fid}")
        ax[0].legend()

        ax[1].bar(x - width / 2, std, width, label="mit Outlier", color=cols)
        ax[1].bar(x + width / 2, std_no_out, width, label="ohne Outlier", color=cols, alpha=0.5)
        ax[1].set_ylabel("Standardabweichung")
        ax[1].set_title(f"Standardabweichung mit/ohne Outlier – ID {fid}")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(xlabels, rotation=30, ha="right")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_GroupedBar_Mean_Std.png"))
        plt.close()
        logging.info(f"[Report] Plot gespeichert: {outdir}/{fid}_GroupedBar_Mean_Std.png")
