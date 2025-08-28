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
from config.plot_config import PlotConfig, PlotOptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PlotService:
    @classmethod
    def overlay_plots(cls, config: PlotConfig, options: PlotOptions) -> None:
        colors = config.ensure_colors()
        os.makedirs(config.path, exist_ok=True)

        # ---- WITH (inkl. Outlier) sammeln ----
        data_with_all: Dict[str, np.ndarray] = {}
        for fid in config.folder_ids:
            data_with, _ = cls._load_data(fid, config.filenames, config.versions)
            if not data_with:
                logging.warning(f"[Report] Keine WITH-Daten für {fid} gefunden.")
                continue
            data_with_all.update(data_with)

        if not data_with_all:
            logging.warning("[Report] Keine Daten gefunden – keine Plots erzeugt.")
            return

        # ---- INLIER (aus *_coordinates_inlier_std.txt) sammeln ----
        data_inlier_all: Dict[str, np.ndarray] = {}
        for fid in config.folder_ids:
            for v in config.versions:
                label = f"{v}_{fid}"
                base_inl = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances_coordinates_inlier_std.txt"
                path_inl = cls._resolve(fid, base_inl)
                logging.info(f"[Report] Lade INLIER: {path_inl}")
                if not os.path.exists(path_inl):
                    logging.warning(f"[Report] Datei fehlt (INLIER): {path_inl}")
                    continue
                try:
                    arr = cls._load_coordinates_inlier_distances(path_inl)
                except Exception as e:
                    logging.error(f"[Report] Laden fehlgeschlagen (INLIER: {path_inl}): {e}")
                    continue
                if arr.size:
                    data_inlier_all[label] = arr

        # Gemeinsamer Range (über WITH, damit Seiten vergleichbar sind)
        data_min, data_max, x = cls._get_common_range(data_with_all)

        # EIN Satz Overlays für ALLE Folder gemeinsam
        fid = "ALLFOLDERS"

        # -------- Seite 1: WITH --------
        fname = "ALL_WITH"
        gauss_with = {k: norm.fit(v) for k, v in data_with_all.items() if v.size}
        if options.plot_hist:
            cls._plot_overlay_histogram(fid, fname, data_with_all, config.bins, data_min, data_max, colors, config.path)
        if options.plot_gauss:
            cls._plot_overlay_gauss(fid, fname, data_with_all, gauss_with, x, colors, config.path)
        if options.plot_weibull:
            cls._plot_overlay_weibull(fid, fname, data_with_all, x, colors, config.path)
        if options.plot_box:
            cls._plot_overlay_boxplot(fid, fname, data_with_all, colors, config.path)
        if options.plot_qq:
            cls._plot_overlay_qq(fid, fname, data_with_all, colors, config.path)
        if options.plot_grouped_bar:
            cls._plot_grouped_bar_means_stds_dual(fid, fname, data_with_all, data_inlier_all, colors, config.path)
        if options.plot_violin:
            cls._plot_overlay_violin(fid, fname, data_with_all, colors, config.path)
        logging.info(f"[Report] PNGs für {fid} (WITH) erzeugt.")

        # -------- Seite 2: INLIER --------
        fname = "ALL_INLIER"
        if data_inlier_all:
            gauss_inl = {k: norm.fit(v) for k, v in data_inlier_all.items() if v.size}
            if options.plot_hist:
                cls._plot_overlay_histogram(fid, fname, data_inlier_all, config.bins, data_min, data_max, colors, config.path)
            if options.plot_gauss:
                cls._plot_overlay_gauss(fid, fname, data_inlier_all, gauss_inl, x, colors, config.path)
            if options.plot_weibull:
                cls._plot_overlay_weibull(fid, fname, data_inlier_all, x, colors, config.path)
            if options.plot_box:
                cls._plot_overlay_boxplot(fid, fname, data_inlier_all, colors, config.path)
            if options.plot_qq:
                cls._plot_overlay_qq(fid, fname, data_inlier_all, colors, config.path)
            if options.plot_grouped_bar:
                cls._plot_grouped_bar_means_stds_dual(fid, fname, data_with_all, data_inlier_all, colors, config.path)
            if options.plot_violin:
                cls._plot_overlay_violin(fid, fname, data_inlier_all, colors, config.path)
            logging.info(f"[Report] PNGs für {fid} (INLIER) erzeugt.")
        else:
            logging.warning("[Report] Keine INLIER-Daten gefunden – zweite Seite bleibt leer.")


    @classmethod
    def summary_pdf(cls, config: PlotConfig) -> None:
        plot_types = [
            ("OverlayHistogramm", "Histogramm", (0, 0)),
            ("Boxplot", "Boxplot", (0, 1)),
            ("OverlayGaussFits", "Gauss-Fit", (0, 2)),
            ("OverlayWeibullFits", "Weibull-Fit", (1, 0)),
            ("QQPlot", "Q-Q-Plot", (1, 1)),
            ("GroupedBar_Mean_Std", "Mittelwert & Std Dev", (1, 2)),
        ]

        fid = "ALLFOLDERS"
        outfile = os.path.join(config.path, f"{fid}_comparison_report.pdf")
        pdf = PdfPages(outfile)

        def _add_page(suffix_label: str, title_suffix: str):
            fig, axs = plt.subplots(2, 3, figsize=(24, 16))
            for suffix, title, (row, col) in plot_types:
                ax = axs[row, col]
                png = os.path.join(config.path, f"{fid}_{suffix_label}_{suffix}.png")
                if os.path.exists(png):
                    img = mpimg.imread(png)
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(title, fontsize=22)
                else:
                    ax.axis("off")
                    ax.set_title(f"{title}\n(nicht gefunden)", fontsize=18)
            plt.suptitle(f"{fid} – Vergleichsplots ({title_suffix})", fontsize=28)
            plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.08, hspace=0.15)
            pdf.savefig(fig)
            plt.close(fig)

        # Seite 1: WITH
        _add_page("ALL_WITH", "inkl. Outlier")

        # Seite 2: INLIER
        _add_page("ALL_INLIER", "ohne Outlier (Inlier)")

        pdf.close()
        logging.info(f"[Report] Zusammenfassung gespeichert: {outfile}")


    # ------- Loader & Helpers --------------------------------

    @staticmethod
    def _resolve(fid: str, filename: str) -> str:
        """
        Unterstützt sowohl '<fid>/<file>' als auch 'data/<fid>/<file>'.
        """
        p1 = os.path.join(fid, filename)
        if os.path.exists(p1):
            return p1
        return os.path.join("data","Multi-illumination", "Job_0378_8400-110", "1-3_2-3", fid, filename)

    @staticmethod
    def _load_1col_distances(path: str) -> np.ndarray:
        """Lädt 1-Spalten Distanzdatei ohne Header."""
        arr = np.loadtxt(path, ndmin=2)        # shape (N,1)
        vals = arr[:, 0].astype(float)
        return vals[np.isfinite(vals)]

    @staticmethod
    def _load_coordinates_inlier_distances(path: str) -> np.ndarray:
        """Lädt 4-Spalten coordinates_inlier_* mit Header; nimmt letzte Spalte als Distanz."""
        # Header vorhanden -> skiprows=1
        arr = np.loadtxt(path, ndmin=2, skiprows=1)  # shape (N,4) erwartet
        if arr.shape[1] < 4:
            raise ValueError(f"Erwarte 4 Spalten (x y z distance) in: {path}")
        vals = arr[:, -1].astype(float)
        return vals[np.isfinite(vals)]


    @classmethod
    def _load_data(cls, fid: str, filenames: List[str], versions: List[str]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Tuple[float, float]]
    ]:
        """
        Rückwärts-kompatibel: liefert weiterhin 'WITH' (inkl. Outlier).
        Für INLIER wird separat in overlay_plots geladen (siehe unten).
        """
        data_with: Dict[str, np.ndarray] = {}
        gauss_with: Dict[str, Tuple[float, float]] = {}

        for v in versions:
            # Deine Dateinamen-Patterns:
            base_with = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances.txt"
            path_with = cls._resolve(fid, base_with)
            logging.info(f"[Report] Lade WITH: {path_with}")

            if not os.path.exists(path_with):
                logging.warning(f"[Report] Datei fehlt (WITH): {path_with}")
                continue

            try:
                # CC könnte auch Semikolon-CSV sein; deine Angabe oben sagt 1 Spalte ohne Header.
                if v.lower() == "cc":
                    # Fallback: zuerst versuchen wir einfach als 1-Spalten-Text:
                    try:
                        arr = cls._load_1col_distances(path_with)
                    except Exception:
                        # Falls es doch CSV ist:
                        df = pd.read_csv(path_with, sep=";")
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        if len(num_cols) == 0:
                            raise ValueError("Keine numerische Spalte gefunden (CC).")
                        arr = df[num_cols[0]].astype(float).to_numpy()
                        arr = arr[np.isfinite(arr)]
                else:
                    arr = cls._load_1col_distances(path_with)
            except Exception as e:
                logging.error(f"[Report] Laden fehlgeschlagen (WITH: {path_with}): {e}")
                continue

            if arr.size:
                label = f"{v}_{fid}"
                data_with[label] = arr
                mu, std = norm.fit(arr)
                gauss_with[label] = (float(mu), float(std))

        return data_with, gauss_with


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
    def _plot_grouped_bar_means_stds_dual(
        fid: str,
        fname: str,
        data_with: Dict[str, np.ndarray],
        data_inlier: Dict[str, np.ndarray],
        colors: Dict[str, str],
        outdir: str,
    ) -> None:
        """
        Pro FOLDER (aus keys wie 'python_1-1') nebeneinander:
        - Balken 'WITH' (mit Outlier)
        - Balken 'INLIER' (ohne Outlier)
        Aggregation: concat über alle Versionen eines Folders.
        """
        # Hilfsfunktion: FOLDER-ID aus Label "version_folder"
        def _folder_of(label: str) -> str:
            # label ist "version_fid" -> wir wollen die komplette fid, auch wenn sie Unterstriche enthält
            return label.split("_", 1)[1] if "_" in label else label

        # Ordne Werte je Folder (concat über Versionen)
        folder_to_with: Dict[str, np.ndarray] = {}
        folder_to_inl: Dict[str, np.ndarray] = {}

        for k, arr in data_with.items():
            f = _folder_of(k)
            folder_to_with.setdefault(f, [])
            folder_to_with[f].append(arr)
        for k, arr in data_inlier.items():
            f = _folder_of(k)
            folder_to_inl.setdefault(f, [])
            folder_to_inl[f].append(arr)

        # Einheitliche Folder-Reihenfolge
        all_folders = sorted(set(folder_to_with.keys()) | set(folder_to_inl.keys()))

        # Kennzahlen je Folder
        means_with, means_inl, stds_with, stds_inl, xlabels, bar_colors = [], [], [], [], [], []
        for f in all_folders:
            arr_with = np.concatenate(folder_to_with.get(f, [])) if f in folder_to_with else np.array([])
            arr_inl  = np.concatenate(folder_to_inl.get(f, [])) if f in folder_to_inl  else np.array([])

            mean_w_signed = float(np.mean(arr_with)) if arr_with.size else np.nan
            std_w         = float(np.std(arr_with))  if arr_with.size else np.nan
            mean_i_signed = float(np.mean(arr_inl))  if arr_inl.size  else np.nan
            std_i         = float(np.std(arr_inl))   if arr_inl.size  else np.nan

            xlabels.append(f)
            mean_w = float(np.abs(mean_w_signed)) if np.isfinite(mean_w_signed) else np.nan
            mean_i = float(np.abs(mean_i_signed)) if np.isfinite(mean_i_signed) else np.nan
            
            means_with.append(mean_w); stds_with.append(std_w)
            means_inl.append(mean_i);  stds_inl.append(std_i)

            # Farbe pro Folder (nimm erste passende Serie, sonst Default)
            # Versuche label "{irgendeine_version}_{folder}" zu finden:
            candidate_label = next((k for k in data_with.keys() if k.endswith("_" + f)), None)
            c = colors.get(candidate_label, "#8aa2ff")
            bar_colors.append(c)

        x = np.arange(len(all_folders))
        width = 0.4

        fig, ax = plt.subplots(2, 1, figsize=(max(10, len(all_folders) * 1.8), 8), sharex=True)

        # Mittelwerte
        ax[0].bar(x - width/2, means_with, width, label="mit Outlier (WITH)", color=bar_colors)
        ax[0].bar(x + width/2, means_inl,  width, label="ohne Outlier (INLIER)", color=bar_colors, alpha=0.55)
        ax[0].set_ylabel("Mittelwert (|μ|)")      # optional klarstellen
        ax[0].set_title(f"Mittelwert je Folder – {fid}/{fname}")
        ax[0].set_ylim(bottom=0)                  # NEU: nie unter 0
        ax[0].legend()

        # Standardabweichungen
        ax[1].bar(x - width/2, stds_with, width, label="mit Outlier (WITH)", color=bar_colors)
        ax[1].bar(x + width/2, stds_inl,  width, label="ohne Outlier (INLIER)", color=bar_colors, alpha=0.55)
        ax[1].set_ylabel("Standardabweichung (σ)")
        ax[1].set_title(f"Standardabweichung je Folder – {fid}/{fname}")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(xlabels, rotation=30, ha="right")
        ax[1].set_ylim(bottom=0)                  # NEU: nie unter 0
        ax[1].legend()

        plt.tight_layout()
        out = os.path.join(outdir, f"{fid}_{fname}_GroupedBar_Mean_Std.png")
        plt.savefig(out)
        plt.close()
        logging.info(f"[Report] Plot gespeichert: {out}")



    @staticmethod
    def _plot_overlay_violin(fid: str, fname: str, data: Dict[str, np.ndarray],
                             colors: Dict[str, str], outdir: str) -> None:
        try:
            import seaborn as sns
            records = [pd.DataFrame({"Version": v, "Distanz": arr}) for v, arr in data.items()]
            if not records:
                return
            df = pd.concat(records, ignore_index=True)
            palette = {v: colors.get(v) for v in df["Version"].unique()}

            plt.figure(figsize=(10, 6))
            sns.violinplot(data=df, x="Version", y="Distanz", palette=palette, cut=0, inner="quartile")
            plt.title(f"Violinplot – {fid}/{fname}")
            plt.xlabel("Version")
            plt.ylabel("M3C2-Distanz")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Violinplot.png"))
            plt.close()
        except Exception as e:
            logging.warning(f"[Report] Violinplot fehlgeschlagen ({fid}/{fname}): {e}")
