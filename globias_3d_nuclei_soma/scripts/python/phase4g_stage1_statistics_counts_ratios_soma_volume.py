# Stage 1 statistics only (no plotting)
# Analyze paired per-field nuclei/soma counts and soma volume distributions.

INPUT_COUNTS_CSV = "/app/data/projects/globias_3d_nuclei_soma/processed_images/batch/batch_per_field_counts_status.csv"
INPUT_SOMA_OBJECTS_CSV = "/app/data/projects/globias_3d_nuclei_soma/processed_images/batch/batch_soma_objects_combined.csv"
OUTPUT_DIR = "/app/data/projects/globias_3d_nuclei_soma/data"

OUTPUT_STATS_CSV = "Statistics_Results.csv"
OUTPUT_COUNTS_CLEAN_CSV = "field_counts_with_safe_ratios.csv"
OUTPUT_MISSINGNESS_CSV = "missingness_report.csv"
OUTPUT_OUTLIERS_CSV = "iqr_outlier_report.csv"
OUTPUT_SOMA_OBJECT_SUMMARY_CSV = "soma_voxel_volume_object_level_summary.csv"
OUTPUT_SOMA_FIELD_SUMMARY_CSV = "soma_field_median_iqr_by_field.csv"

IQR_K = 1.5
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42
EXPECTED_N_FIELDS = 20


def ensure_columns(df, required_cols, df_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"VERIFICATION FAILED: missing columns in {df_name}: {missing}")


def iqr_bounds_and_outliers(series, k=1.5):
    arr = pd.Series(series).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    n_out = int(((arr < lo) | (arr > hi)).sum())
    return float(lo), float(hi), n_out


def describe_numeric(series):
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
        }

    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if s.size > 1 else np.nan,
        "median": float(s.median()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "min": float(s.min()),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
    }


def bootstrap_median_ci(series, n_boot=10000, seed=42, ci=95.0):
    s = pd.Series(series).dropna().astype(float).to_numpy()
    if s.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(s, size=s.size, replace=True)
        meds[i] = np.median(sample)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(meds, alpha))
    hi = float(np.percentile(meds, 100.0 - alpha))
    return lo, hi


def pearson_ci_fisher_z(r, n, ci=0.95):
    if np.isnan(r) or n < 4:
        return np.nan, np.nan
    if abs(r) >= 1.0:
        return float(r), float(r)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(0.5 + ci / 2.0)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def add_result(results, **kwargs):
    base = {
        "analysis_id": "",
        "analysis_category": "",
        "dataset_subset": "",
        "variable": "",
        "test_name": "",
        "estimate_type": "",
        "estimate": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "ci_level": "95%",
        "p_value": np.nan,
        "n_fields": np.nan,
        "n_objects": np.nan,
        "mean": np.nan,
        "sd": np.nan,
        "median": np.nan,
        "iqr": np.nan,
        "min": np.nan,
        "max": np.nan,
        "units": "",
        "analysis_level": "",
        "interpretation": "",
        "notes": "",
    }
    base.update(kwargs)
    results.append(base)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    counts_df = pd.read_csv(INPUT_COUNTS_CSV)
    soma_df = pd.read_csv(INPUT_SOMA_OBJECTS_CSV)

    print("Counts table columns:", list(counts_df.columns), flush=True)
    print("Soma object table columns:", list(soma_df.columns), flush=True)

    required_counts_cols = [
        "field_id",
        "dapi_total_nuclei_count",
        "soma_total_object_count",
        "dapi_nonborder_nuclei_count",
        "soma_nonborder_object_count",
        "field_status",
    ]
    required_soma_cols = [
        "field_id",
        "voxel_volume",
        "touches_any_border",
        "provisional_volume_um3",
    ]
    ensure_columns(counts_df, required_counts_cols, "counts_df")
    ensure_columns(soma_df, required_soma_cols, "soma_df")

    # Missingness report
    missing_rows = []
    for src_name, df in (("counts", counts_df), ("soma_objects", soma_df)):
        n_rows = len(df)
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            missing_rows.append({
                "source": src_name,
                "column": col,
                "n_rows": n_rows,
                "n_missing": n_miss,
                "pct_missing": (100.0 * n_miss / n_rows) if n_rows > 0 else np.nan,
            })
    missingness_df = pd.DataFrame(missing_rows)
    missing_path = os.path.join(OUTPUT_DIR, OUTPUT_MISSINGNESS_CSV)
    missingness_df.to_csv(missing_path, index=False)

    # Verify expected field count and status completeness
    n_fields = int(counts_df["field_id"].nunique())
    if n_fields != EXPECTED_N_FIELDS:
        raise ValueError(
            f"VERIFICATION FAILED: expected {EXPECTED_N_FIELDS} unique fields, found {n_fields}."
        )

    if len(counts_df) != EXPECTED_N_FIELDS:
        raise ValueError(
            f"VERIFICATION FAILED: expected {EXPECTED_N_FIELDS} rows in counts table, found {len(counts_df)}."
        )

    status_ok = counts_df["field_status"].astype(str).str.lower().eq("ok")
    n_not_ok = int((~status_ok).sum())
    if n_not_ok > 0:
        bad_fields = counts_df.loc[~status_ok, "field_id"].tolist()
        raise ValueError(
            f"VERIFICATION FAILED: {n_not_ok} field_status values are not 'ok'. Fields: {bad_fields}"
        )

    # Safe ratio recomputation (zero-denominator guarded)
    counts_df = counts_df.copy()
    total_denom = counts_df["soma_total_object_count"].astype(float)
    nonborder_denom = counts_df["soma_nonborder_object_count"].astype(float)

    counts_df["nuclei_to_soma_ratio_total_safe"] = np.where(
        total_denom > 0,
        counts_df["dapi_total_nuclei_count"].astype(float) / total_denom,
        np.nan,
    )
    counts_df["nuclei_to_soma_ratio_nonborder_safe"] = np.where(
        nonborder_denom > 0,
        counts_df["dapi_nonborder_nuclei_count"].astype(float) / nonborder_denom,
        np.nan,
    )

    n_zero_total = int((total_denom <= 0).sum())
    n_zero_nonborder = int((nonborder_denom <= 0).sum())
    print(f"Zero denominator fields (total ratio): {n_zero_total}", flush=True)
    print(f"Zero denominator fields (nonborder ratio): {n_zero_nonborder}", flush=True)
    if n_zero_total > 0 or n_zero_nonborder > 0:
        print("WARNING: some ratios set to NaN due to zero soma counts.", flush=True)

    counts_clean_path = os.path.join(OUTPUT_DIR, OUTPUT_COUNTS_CLEAN_CSV)
    counts_df.to_csv(counts_clean_path, index=False)

    # IQR outlier reporting (detected, not removed)
    outlier_targets = [
        ("dapi_total_nuclei_count", counts_df["dapi_total_nuclei_count"], "objects_per_field"),
        ("soma_total_object_count", counts_df["soma_total_object_count"], "objects_per_field"),
        ("dapi_nonborder_nuclei_count", counts_df["dapi_nonborder_nuclei_count"], "objects_per_field"),
        ("soma_nonborder_object_count", counts_df["soma_nonborder_object_count"], "objects_per_field"),
        ("nuclei_to_soma_ratio_total_safe", counts_df["nuclei_to_soma_ratio_total_safe"], "ratio"),
        ("nuclei_to_soma_ratio_nonborder_safe", counts_df["nuclei_to_soma_ratio_nonborder_safe"], "ratio"),
    ]

    outlier_rows = []
    for name, series, units in outlier_targets:
        lo, hi, n_out = iqr_bounds_and_outliers(series, k=IQR_K)
        outlier_rows.append({
            "variable": name,
            "iqr_k": IQR_K,
            "lower_fence": lo,
            "upper_fence": hi,
            "n_outliers": n_out,
            "n_nonmissing": int(pd.Series(series).notna().sum()),
            "units": units,
        })
        print(f"IQR outliers (k={IQR_K}) for {name}: {n_out}", flush=True)

    outlier_df = pd.DataFrame(outlier_rows)
    outlier_path = os.path.join(OUTPUT_DIR, OUTPUT_OUTLIERS_CSV)
    outlier_df.to_csv(outlier_path, index=False)

    results = []

    # Descriptive count stats across fields
    count_specs = [
        ("dapi_total_nuclei_count", "total_nuclei_count", "all_objects", "objects_per_field"),
        ("soma_total_object_count", "total_soma_count", "all_objects", "objects_per_field"),
        ("dapi_nonborder_nuclei_count", "nonborder_nuclei_count", "nonborder_objects", "objects_per_field"),
        ("soma_nonborder_object_count", "nonborder_soma_count", "nonborder_objects", "objects_per_field"),
    ]

    for col, variable_name, subset, units in count_specs:
        d = describe_numeric(counts_df[col])
        ci_lo, ci_hi = bootstrap_median_ci(counts_df[col], n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        add_result(
            results,
            analysis_id=f"desc_{variable_name}",
            analysis_category="descriptive_counts",
            dataset_subset=subset,
            variable=variable_name,
            test_name="descriptive_with_bootstrap_median_ci",
            estimate_type="median",
            estimate=d["median"],
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=np.nan,
            n_fields=d["n"],
            n_objects=float(counts_df[col].sum()),
            mean=d["mean"],
            sd=d["sd"],
            median=d["median"],
            iqr=d["iqr"],
            min=d["min"],
            max=d["max"],
            units=units,
            analysis_level="field",
            interpretation=f"Across {d['n']} paired fields, {variable_name} shows this central tendency and spread.",
            notes=f"Bootstrap median CI with {BOOTSTRAP_N} resamples; IQR outliers reported separately (k={IQR_K}).",
        )

    # Correlation analyses (field-level, paired)
    def run_corr(x_col, y_col, subset_label, analysis_id_prefix):
        pair_df = counts_df[["field_id", x_col, y_col]].dropna().copy()
        if len(pair_df) < 3:
            raise ValueError(
                f"VERIFICATION FAILED: fewer than 3 paired rows for {x_col} vs {y_col}."
            )

        x = pair_df[x_col].to_numpy(dtype=float)
        y = pair_df[y_col].to_numpy(dtype=float)

        pear = stats.pearsonr(x, y)
        r = float(pear.statistic)
        p_pear = float(pear.pvalue)

        try:
            ci_obj = pear.confidence_interval(confidence_level=0.95)
            r_lo, r_hi = float(ci_obj.low), float(ci_obj.high)
        except Exception:
            r_lo, r_hi = pearson_ci_fisher_z(r, len(pair_df), ci=0.95)

        spear = stats.spearmanr(x, y, nan_policy="omit")
        rho = float(spear.statistic)
        p_spear = float(spear.pvalue)

        print(
            f"{subset_label} Pearson: r={r:.6f}, p={p_pear:.6g}, 95% CI=[{r_lo:.6f}, {r_hi:.6f}]",
            flush=True,
        )
        print(
            f"{subset_label} Spearman: rho={rho:.6f}, p={p_spear:.6g}",
            flush=True,
        )

        add_result(
            results,
            analysis_id=f"{analysis_id_prefix}_pearson",
            analysis_category="association",
            dataset_subset=subset_label,
            variable=f"{x_col}~{y_col}",
            test_name="Pearson correlation",
            estimate_type="pearson_r",
            estimate=r,
            ci_lower=r_lo,
            ci_upper=r_hi,
            p_value=p_pear,
            n_fields=len(pair_df),
            n_objects=np.nan,
            units="correlation",
            analysis_level="field",
            interpretation="Linear association between paired per-field counts.",
            notes="Field is the biological replicate; object-level pooling avoided.",
        )

        add_result(
            results,
            analysis_id=f"{analysis_id_prefix}_spearman",
            analysis_category="association",
            dataset_subset=subset_label,
            variable=f"{x_col}~{y_col}",
            test_name="Spearman rank correlation",
            estimate_type="spearman_rho",
            estimate=rho,
            ci_lower=np.nan,
            ci_upper=np.nan,
            p_value=p_spear,
            n_fields=len(pair_df),
            n_objects=np.nan,
            units="correlation",
            analysis_level="field",
            interpretation="Monotonic association robust to outliers/nonlinearity.",
            notes="Sensitivity complement to Pearson.",
        )

    run_corr(
        "dapi_total_nuclei_count",
        "soma_total_object_count",
        "total_counts",
        "corr_total_counts",
    )
    run_corr(
        "dapi_nonborder_nuclei_count",
        "soma_nonborder_object_count",
        "nonborder_counts",
        "corr_nonborder_counts",
    )

    # Ratio summaries across fields
    ratio_specs = [
        ("nuclei_to_soma_ratio_total_safe", "nuclei_to_soma_ratio_total", "total_counts"),
        ("nuclei_to_soma_ratio_nonborder_safe", "nuclei_to_soma_ratio_nonborder", "nonborder_counts"),
    ]

    for col, variable_name, subset in ratio_specs:
        d = describe_numeric(counts_df[col])
        ci_lo, ci_hi = bootstrap_median_ci(counts_df[col], n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        add_result(
            results,
            analysis_id=f"ratio_{variable_name}",
            analysis_category="ratio_summary",
            dataset_subset=subset,
            variable=variable_name,
            test_name="descriptive_with_bootstrap_median_ci",
            estimate_type="median_ratio",
            estimate=d["median"],
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            p_value=np.nan,
            n_fields=d["n"],
            n_objects=np.nan,
            mean=d["mean"],
            sd=d["sd"],
            median=d["median"],
            iqr=d["iqr"],
            min=d["min"],
            max=d["max"],
            units="nuclei_per_soma_object",
            analysis_level="field",
            interpretation="Per-field nuclei-to-soma ratio summary (paired field replicate level).",
            notes=f"Zero denominators converted to NaN; median CI from {BOOTSTRAP_N} field-level bootstrap resamples.",
        )

    # Soma object-level volume summary (descriptive only; not inferential replicate level)
    soma_summary_rows = []
    subset_masks = {
        "all_objects": pd.Series([True] * len(soma_df), index=soma_df.index),
        "nonborder_objects": ~soma_df["touches_any_border"].astype(bool),
    }

    all_field_ids = counts_df["field_id"].astype(str).tolist()
    per_field_rows = []

    for subset_name, mask in subset_masks.items():
        sub = soma_df.loc[mask].copy()
        vol = sub["voxel_volume"].astype(float)
        d = describe_numeric(vol)

        quantiles = {}
        for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            quantiles[f"q{q:02d}"] = float(np.percentile(vol, q)) if len(vol) else np.nan

        soma_summary_rows.append({
            "subset": subset_name,
            "n_objects": d["n"],
            "n_fields_with_objects": int(sub["field_id"].nunique()),
            "mean_voxel_volume": d["mean"],
            "sd_voxel_volume": d["sd"],
            "median_voxel_volume": d["median"],
            "iqr_voxel_volume": d["iqr"],
            "min_voxel_volume": d["min"],
            "max_voxel_volume": d["max"],
            **quantiles,
        })

        add_result(
            results,
            analysis_id=f"soma_object_volume_{subset_name}",
            analysis_category="soma_volume_object_level",
            dataset_subset=subset_name,
            variable="soma_voxel_volume",
            test_name="descriptive_quantiles",
            estimate_type="median_voxel_volume",
            estimate=d["median"],
            ci_lower=np.nan,
            ci_upper=np.nan,
            p_value=np.nan,
            n_fields=int(sub["field_id"].nunique()),
            n_objects=d["n"],
            mean=d["mean"],
            sd=d["sd"],
            median=d["median"],
            iqr=d["iqr"],
            min=d["min"],
            max=d["max"],
            units="voxels",
            analysis_level="object",
            interpretation="Primary endpoint in voxel units; object-level distribution is descriptive.",
            notes="Objects are nested within fields; no object-level inferential testing performed.",
        )

        # Provisional um3 summary (explicitly labeled secondary)
        if "provisional_volume_um3" in sub.columns:
            prov = sub["provisional_volume_um3"].astype(float)
            pdsc = describe_numeric(prov)
            add_result(
                results,
                analysis_id=f"soma_object_volume_um3_provisional_{subset_name}",
                analysis_category="soma_volume_object_level_provisional",
                dataset_subset=subset_name,
                variable="soma_provisional_volume_um3",
                test_name="descriptive_quantiles",
                estimate_type="median_provisional_um3",
                estimate=pdsc["median"],
                ci_lower=np.nan,
                ci_upper=np.nan,
                p_value=np.nan,
                n_fields=int(sub["field_id"].nunique()),
                n_objects=pdsc["n"],
                mean=pdsc["mean"],
                sd=pdsc["sd"],
                median=pdsc["median"],
                iqr=pdsc["iqr"],
                min=pdsc["min"],
                max=pdsc["max"],
                units="provisional_um3",
                analysis_level="object",
                interpretation="Secondary/provisional physical-volume summary only.",
                notes="PROVISIONAL: XY=0.324512482 um inferred and Z=1.0 um assumed; not a primary endpoint.",
            )

        # Per-field medians/IQRs for replicate-level volume distribution
        grouped = sub.groupby("field_id")["voxel_volume"]
        gdf = grouped.agg(
            n_soma_objects="size",
            soma_voxel_median="median",
            soma_voxel_q1=lambda s: np.percentile(s, 25),
            soma_voxel_q3=lambda s: np.percentile(s, 75),
        ).reset_index()
        gdf["soma_voxel_iqr"] = gdf["soma_voxel_q3"] - gdf["soma_voxel_q1"]
        gdf["subset"] = subset_name

        # Ensure all fields represented in per-field output, even if zero objects in subset
        gdf = pd.DataFrame({"field_id": all_field_ids}).merge(gdf, on="field_id", how="left")
        gdf["subset"] = subset_name
        gdf["n_soma_objects"] = gdf["n_soma_objects"].fillna(0).astype(int)

        per_field_rows.append(gdf)

        # Across-field distribution of field medians (replicate-level)
        field_meds = gdf.loc[gdf["n_soma_objects"] > 0, "soma_voxel_median"]
        fd = describe_numeric(field_meds)
        f_lo, f_hi = bootstrap_median_ci(field_meds, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        lo_out, hi_out, n_out = iqr_bounds_and_outliers(field_meds, k=IQR_K)
        print(
            f"IQR outliers (k={IQR_K}) for field medians [{subset_name}]: {n_out}",
            flush=True,
        )

        add_result(
            results,
            analysis_id=f"field_median_soma_volume_{subset_name}",
            analysis_category="soma_volume_field_level",
            dataset_subset=subset_name,
            variable="field_median_soma_voxel_volume",
            test_name="descriptive_with_bootstrap_median_ci",
            estimate_type="median_of_field_medians",
            estimate=fd["median"],
            ci_lower=f_lo,
            ci_upper=f_hi,
            p_value=np.nan,
            n_fields=fd["n"],
            n_objects=np.nan,
            mean=fd["mean"],
            sd=fd["sd"],
            median=fd["median"],
            iqr=fd["iqr"],
            min=fd["min"],
            max=fd["max"],
            units="voxels",
            analysis_level="field",
            interpretation="Replicate-level soma-size distribution uses field medians, not pooled objects.",
            notes=f"Median CI from {BOOTSTRAP_N} bootstrap resamples over field medians; field-median outliers (IQR k={IQR_K})={n_out}.",
        )

    soma_object_summary_df = pd.DataFrame(soma_summary_rows)
    soma_object_summary_path = os.path.join(OUTPUT_DIR, OUTPUT_SOMA_OBJECT_SUMMARY_CSV)
    soma_object_summary_df.to_csv(soma_object_summary_path, index=False)

    soma_field_summary_df = pd.concat(per_field_rows, ignore_index=True)
    soma_field_summary_path = os.path.join(OUTPUT_DIR, OUTPUT_SOMA_FIELD_SUMMARY_CSV)
    soma_field_summary_df.to_csv(soma_field_summary_path, index=False)

    # Assemble Statistics_Results.csv
    stats_df = pd.DataFrame(results)
    stats_path = os.path.join(OUTPUT_DIR, OUTPUT_STATS_CSV)
    stats_df.to_csv(stats_path, index=False)

    # Verification checks (definitional invariants only)
    expected_stats_cols = [
        "analysis_id", "analysis_category", "dataset_subset", "variable", "test_name",
        "estimate_type", "estimate", "ci_lower", "ci_upper", "p_value", "n_fields",
        "n_objects", "mean", "sd", "median", "iqr", "min", "max", "units",
        "analysis_level", "interpretation", "notes"
    ]
    ensure_columns(stats_df, expected_stats_cols, "Statistics_Results")

    if len(counts_df) > 0 and len(stats_df) == 0:
        raise ValueError("VERIFICATION FAILED: input had rows but Statistics_Results is empty.")

    for p in stats_df["p_value"].dropna().astype(float):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"VERIFICATION FAILED: p-value out of [0,1]: {p}")

    for corr_val in stats_df.loc[
        stats_df["estimate_type"].isin(["pearson_r", "spearman_rho"]), "estimate"
    ].dropna().astype(float):
        if not (-1.0000001 <= corr_val <= 1.0000001):
            raise ValueError(f"VERIFICATION FAILED: correlation outside [-1,1]: {corr_val}")

    ci_check = stats_df[["ci_lower", "ci_upper"]].dropna()
    if not ci_check.empty:
        bad_ci = (ci_check["ci_lower"] > ci_check["ci_upper"]).any()
        if bad_ci:
            raise ValueError("VERIFICATION FAILED: some CI lower bounds exceed upper bounds.")

    nonneg_cols = [
        "dapi_total_nuclei_count", "soma_total_object_count",
        "dapi_nonborder_nuclei_count", "soma_nonborder_object_count",
    ]
    for col in nonneg_cols:
        if (counts_df[col] < 0).any():
            raise ValueError(f"VERIFICATION FAILED: negative counts found in {col}")

    output_paths = [
        stats_path,
        counts_clean_path,
        missing_path,
        outlier_path,
        soma_object_summary_path,
        soma_field_summary_path,
    ]
    for op in output_paths:
        if not os.path.exists(op) or os.path.getsize(op) == 0:
            raise ValueError(f"VERIFICATION FAILED: expected output missing or empty: {op}")

    # Required explicit key result prints
    corr_rows = stats_df[stats_df["analysis_category"].eq("association")]
    for _, row in corr_rows.iterrows():
        print(
            f"KEY RESULT | {row['analysis_id']} | {row['test_name']} | "
            f"stat={row['estimate']:.6f} | p={row['p_value']:.6g}",
            flush=True,
        )

    print(f"Wrote statistics table: {stats_path}", flush=True)
    print("Stage 1 complete: no plotting performed.", flush=True)


if __name__ == "__main__":
    main()
