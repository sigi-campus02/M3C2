from statistics_service_comparedistances import StatisticsCompareDistances

# "0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"

folder_ids = ["0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
ref_variants = ["ref_ai", "ref"]
outdir = "Plots_BlandAltman"

def main():
    for fid in folder_ids:
        StatisticsCompareDistances.bland_altman_plot(
            folder_ids=[fid],
            ref_variants=ref_variants,
            outdir=outdir,
        )


if __name__ == "__main__":
    main()

