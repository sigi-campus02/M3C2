import numpy as np
import os
from statistics_service_comparedistances import StatisticsCompareDistances

# Ordner
folder = "Plots_BlandAltman_Test"
plot_outdir = folder
os.makedirs(folder, exist_ok=True)

# Basiswerte (simulierte Distanzen) – gemeinsamer Zusammenhang für alle Varianten
np.random.seed(42)
n = 1000
base = np.random.normal(0, 1.0, n)

# Hilfsfunktion: speichert ein Paar GENAU unter den später genutzten Variantenamen
def save_pair(var_a, var_b, arr_a, arr_b):
    np.savetxt(os.path.join(folder, f"python_{var_a}_m3c2_distances.txt"), arr_a)
    np.savetxt(os.path.join(folder, f"python_{var_b}_m3c2_distances.txt"), arr_b)

# 1) Perfekte Übereinstimmung
save_pair("ref", "ref_ai", base, base)

# 2) Konstanter Bias (+0.2)
save_pair("ref_bias", "ref_ai_bias", base, base + 0.2)

# 3) Variabler Bias (linear zum Wert): B = A + 0.1*A
save_pair("ref_trend", "ref_ai_trend", base, base + 0.1 * base)

# 4) Reines Rauschen (keine systematische Verschiebung)
sigma_noise = 0.25
save_pair("ref_noise", "ref_ai_noise", base, base + np.random.normal(0.0, sigma_noise, n))

# 5) Proportionaler Bias / Skalierung  B = 1.2 * A
save_pair("ref_scale", "ref_ai_scale", base, 1.2 * base)

# 6) Nichtlinearer Unterschied  B = A + 0.1 * A^2
save_pair("ref_nonlinear", "ref_ai_nonlinear", base, base + 0.1 * (base ** 2))

# 7) Heteroskedastisches Rauschen (Streuung wächst mit |A|)
sigma_hetero = 0.1
save_pair("ref_hetero", "ref_ai_hetero", base, base + np.random.normal(0.0, sigma_hetero, n) * np.abs(base))

# 8) Ausreißer – 5% Punkte bekommen großen Fehler
outlier_rate = 0.05
k = int(n * outlier_rate)
idx = np.random.choice(n, k, replace=False)
b_out = base.copy()
b_out[idx] += 5.0
save_pair("ref_outliers", "ref_ai_outliers", base, b_out)

# 9) Kombination: kleiner Bias + Skalierung + Rauschen + wenige Ausreißer
b_combo = 0.1 + 1.05 * base + np.random.normal(0.0, 0.15, n)
idx2 = np.random.choice(n, int(n * 0.02), replace=False)
b_combo[idx2] -= 3.0
save_pair("ref_combo", "ref_ai_combo", base, b_combo)

# ============ Plots erzeugen ============
def ba(pair):
    StatisticsCompareDistances.bland_altman_plot(
        folder_ids=[folder],
        ref_variants=pair,
        outdir=plot_outdir
    )

ba(["ref", "ref_ai"])
ba(["ref_bias", "ref_ai_bias"])
ba(["ref_trend", "ref_ai_trend"])
ba(["ref_noise", "ref_ai_noise"])
ba(["ref_scale", "ref_ai_scale"])
ba(["ref_nonlinear", "ref_ai_nonlinear"])
ba(["ref_hetero", "ref_ai_hetero"])
ba(["ref_outliers", "ref_ai_outliers"])
ba(["ref_combo", "ref_ai_combo"])

print(f"Fertig. PNGs unter: {plot_outdir}")


import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# --------------------------------------------------------------------------------------
# Pfad-Helfer (kompatibel zu deiner Logik)
def _resolve(folder: str, basename: str) -> str:
    p1 = os.path.join(folder, basename)
    if os.path.exists(p1):
        return p1
    return os.path.join("data", folder, basename)


# Szenario-spezifische Erklärungen (pro Paar)
SCENARIO_NOTES = {
    ("ref", "ref_ai"): (
        "Perfekte Übereinstimmung",
        "B wurde ident mit A gesetzt: B = A (keine Änderungen).",
        "Bias ≈ 0, sehr schmale/keine LoA. Punkte liegen auf y=0 → Methoden sind austauschbar."
    ),
    ("ref_bias", "ref_ai_bias"): (
        "Konstanter Bias",
        "Konstanter Offset von +0.2: B = A + 0.2.",
        "Bias ≈ −0.2 (weil A−B). Punkte parallel zur x‑Achse um den Bias verschoben, LoA schmal."
    ),
    ("ref_trend", "ref_ai_trend"): (
        "Variabler Bias (linear)",
        "Proportionale Verschiebung: B = A + 0.1·A (Bias wächst mit A).",
        "Trend im Plot: Differenzen hängen vom Mittelwert ab (Schräge/„Bauch“). LoA moderat."
    ),
    ("ref_noise", "ref_ai_noise"): (
        "Reines Rauschen",
        "Zufallsrauschen ohne systematische Verschiebung: B = A + ε, ε~N(0, 0.25).",
        "Bias ≈ 0, aber breitere LoA. Punkte zufällig um 0 verteilt → gleiche Lage, geringere Präzision."
    ),
    ("ref_scale", "ref_ai_scale"): (
        "Proportionaler Bias (Skalierung)",
        "Skalierungsfehler: B = 1.2·A (alle Werte 20% größer).",
        "Starker Trend: Differenzen steigen mit dem Mittelwert. Bias kann nahe 0 sein, LoA breit; Austauschbarkeit schlecht."
    ),
    ("ref_nonlinear", "ref_ai_nonlinear"): (
        "Nichtlinearer Unterschied",
        "Nichtlinearer Term: B = A + 0.1·A² (stärkere Abweichungen für große |A|).",
        "Gebogene/gefächerte Struktur; LoA wachsen zu den Rändern → systematische, nichtlineare Differenzen."
    ),
    ("ref_hetero", "ref_ai_hetero"): (
        "Heteroskedastisches Rauschen",
        "Rauschstärke wächst mit |A|: B = A + ε·|A|, ε~N(0,0.1).",
        "Klassische Fächerform: kleine Mittelwerte eng, große breit. Bias ≈ 0, LoA nehmen mit x zu."
    ),
    ("ref_outliers", "ref_ai_outliers"): (
        "Ausreißer",
        "5% der Punkte als starke positive Ausreißer: B_out[idx] += 5.",
        "Wenig Punkte weit außerhalb der LoA. Bias kann kaum verschoben sein, aber Ausreißer dominieren Extrembereiche."
    ),
    ("ref_combo", "ref_ai_combo"): (
        "Kombination mehrerer Fehler",
        "B = 0.1 + 1.05·A + N(0,0.15); zusätzlich 2% starke negative Ausreißer.",
        "Bias ≠ 0, Trend (Skalierung) + erhöhte Streuung + Ausreißer. Austauschbarkeit nur bedingt gegeben."
    ),
}



# Bland–Altman + Report
def bland_altman_report_pdf(
    folder: str,
    pairs: list[tuple[str, str]],
    out_pdf: str = None,
    title: str = None,
    tolerance: float = 0.01,
) -> str:
    """
    Erzeugt ein PDF mit je Seite:
      - Bland–Altman Plot (A vs B)
      - Erklärungstext + Kennzahlen
    Erwartete Dateinamen im Ordner:
      python_{variant}_m3c2_distances.txt
    Beispiel:
      pairs = [("ref","ref_ai"), ("ref_bias","ref_ai_bias"), ...]
    """
    if out_pdf is None:
        out_pdf = os.path.join(folder, "BlandAltman_Report.pdf")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    if title is None:
        title = f"Bland–Altman Report – {folder}"

    created_pages = 0
    with PdfPages(out_pdf) as pdf:

        # Deckblatt
        fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
        fig.suptitle(title, fontsize=18, y=0.92)
        txt = (
            f"Generiert am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Dieser Report vergleicht jeweils zwei Distanz-Outputs per Bland–Altman:\n"
            f"- x-Achse: Mittelwert der beiden Messungen je Punkt\n"
            f"- y-Achse: Differenz (A − B)\n"
            f"Die rote Linie zeigt den systematischen Unterschied (Bias).\n"
            f"Die grünen Linien zeigen die Limits of Agreement (±1.96·SD der Differenzen).\n\n"
            f"Zusätzlich werden pro Seite wichtige Kennzahlen zusammengefasst (Bias, SD, LoA, RMSE, MAE,\n"
            f"Anteil innerhalb ±{tolerance:g}, Anzahl gültiger Punkte und NaN-Quote)."
        )
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.02, 0.95, txt, va="top", ha="left", fontsize=12, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        created_pages += 1

        # Seiten für jedes Paar
        for var_a, var_b in pairs:
            path_a = _resolve(folder, f"python_{var_a}_m3c2_distances.txt")
            path_b = _resolve(folder, f"python_{var_b}_m3c2_distances.txt")

            if not os.path.exists(path_a) or not os.path.exists(path_b):
                # Seite mit Fehlermeldung
                fig = plt.figure(figsize=(11.7, 8.3))
                fig.suptitle(f"{var_a} vs {var_b}", fontsize=16, y=0.92)
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(
                    0.02, 0.95,
                    f"Datei nicht gefunden:\n- {path_a}\n- {path_b}\n\nSeite übersprungen.",
                    va="top", ha="left", fontsize=12, family="monospace", color="crimson"
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                created_pages += 1
                continue

            a_raw = np.loadtxt(path_a)
            b_raw = np.loadtxt(path_b)
            mask = np.isfinite(a_raw) & np.isfinite(b_raw)
            a = a_raw[mask]
            b = b_raw[mask]

            total = max(len(a_raw), len(b_raw))
            valid_n = a.size
            nan_n = total - valid_n
            nan_ratio = nan_n / total if total > 0 else 0.0

            if valid_n == 0:
                fig = plt.figure(figsize=(11.7, 8.3))
                fig.suptitle(f"{var_a} vs {var_b}", fontsize=16, y=0.92)
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(
                    0.02, 0.95,
                    "Alle Werte sind NaN oder leer – keine Auswertung möglich.",
                    va="top", ha="left", fontsize=12, family="monospace", color="crimson"
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                created_pages += 1
                continue

            # Bland–Altman
            mean_vals = (a + b) / 2.0
            diff_vals = a - b
            bias = float(np.mean(diff_vals))
            sd = float(np.std(diff_vals, ddof=1)) if valid_n > 1 else 0.0
            upper = bias + 1.96 * sd
            lower = bias - 1.96 * sd

            # Zusatzmetriken
            rmse = float(np.sqrt(np.mean((diff_vals) ** 2)))
            mae = float(np.mean(np.abs(diff_vals)))
            within_tol = float(np.mean(np.abs(diff_vals) <= tolerance))

            # Plot + Text nebeneinander
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.suptitle(f"{folder}: {var_a} vs {var_b}", fontsize=16, y=0.95)

            # Layout
            gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.5, 1.5], wspace=0.25)

            # --- Plot links ---
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.scatter(mean_vals, diff_vals, s=8, alpha=0.35)
            ax1.axhline(bias, linestyle="--", label=f"Bias = {bias:.4f}")
            ax1.axhline(upper, linestyle="--", label=f"+1.96 SD = {upper:.4f}")
            ax1.axhline(lower, linestyle="--", label=f"-1.96 SD = {lower:.4f}")
            ax1.set_xlabel("Mittelwert der Messungen (A & B)")
            ax1.set_ylabel("Differenz (A − B)")
            ax1.legend(loc="lower left", fontsize=9)
            ax1.grid(alpha=0.2)

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis("off")

            # Allgemeine Erklärung
            general_explain = textwrap.dedent(f"""
                • Der Bland–Altman Plot zeigt pro Punkt den Mittelwert (x) und die Differenz A−B (y).
                • Die rote Linie ist der Bias (mittlere Differenz): systematischer Unterschied.
                • Die grünen Linien sind die Limits of Agreement (Bias ± 1.96·SD): typische Streuung.
                • Zufällige Streuung um den Bias → gute Übereinstimmung.
                • Trend/„Fächer“ → Abhängigkeit der Differenz von der Messgröße (Heteroskedastizität).
            """).strip()

            # Szenario-spezifisch
            title_note, changed_note, interpret_note = SCENARIO_NOTES.get(
                (var_a, var_b),
                (
                    "Szenario",
                    "Keine individuelle Beschreibung hinterlegt.",
                    "Interpretation anhand Bias/LoA/Struktur im Plot."
                )
            )

            summary = (
                f"Anzahl gültig: {valid_n:,}   |   NaN-Quote: {nan_ratio:.1%}\n"
                f"Bias (Mean diff): {bias:.6f}\n"
                f"SD der Differenzen: {sd:.6f}\n"
                f"LoA: [{lower:.6f}, {upper:.6f}]\n"
                f"RMSE: {rmse:.6f}   |   MAE: {mae:.6f}\n"
                f"Anteil |A−B| ≤ {tolerance:g}: {within_tol:.1%}"
            )


            # Blöcke rendern
            ax2.text(0.02, 0.98, title_note, va="top", ha="left", fontsize=12, weight="bold")
            ax2.text(0.02, 0.93, f"Was geändert?\n{changed_note}", va="top", ha="left", fontsize=11)
            ax2.text(0.02, 0.73, f"Interpretation\n{interpret_note}", va="top", ha="left", fontsize=11)
            ax2.text(0.02, 0.53, "Kennzahlen (A−B)", va="top", ha="left", fontsize=12, weight="bold")
            ax2.text(0.02, 0.48, summary, va="top", ha="left", fontsize=11, family="monospace")
            ax2.text(0.02, 0.29, "Allgemeine Erklärung", va="top", ha="left", fontsize=12, weight="bold")
            ax2.text(0.02, 0.24, general_explain, va="top", ha="left", fontsize=11)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            created_pages += 1

    return out_pdf if created_pages > 0 else None


# ===================== Beispielaufruf =====================
if __name__ == "__main__":
    folder = "Plots_BlandAltman_Test"  # hier liegen deine python_{variant}_m3c2_distances.txt
    pairs = [
        ("ref", "ref_ai"),
        ("ref_bias", "ref_ai_bias"),
        ("ref_trend", "ref_ai_trend"),
        ("ref_noise", "ref_ai_noise"),
        ("ref_scale", "ref_ai_scale"),
        ("ref_nonlinear", "ref_ai_nonlinear"),
        ("ref_hetero", "ref_ai_hetero"),
        ("ref_outliers", "ref_ai_outliers"),
        ("ref_combo", "ref_ai_combo"),
    ]
    out = bland_altman_report_pdf(
        folder=folder,
        pairs=pairs,
        out_pdf=os.path.join(folder, "BlandAltman_Report.pdf"),
        title="Bland–Altman Vergleich – Testvarianten",
        tolerance=0.01,
    )
    print("Report geschrieben:", out)

