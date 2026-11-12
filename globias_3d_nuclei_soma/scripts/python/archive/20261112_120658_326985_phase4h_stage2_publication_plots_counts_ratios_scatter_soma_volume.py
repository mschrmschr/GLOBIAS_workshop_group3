# Stage 2 plotting only: reads precomputed statistics and renders publication-quality figures.

STATS_CSV = r'/app/data/projects/globias_3d_nuclei_soma/data/Statistics_Results.csv'
COUNTS_CSV = r'/app/data/projects/globias_3d_nuclei_soma/data/field_counts_with_safe_ratios.csv'
FIELD_SOMA_SUMMARY_CSV = r'/app/data/projects/globias_3d_nuclei_soma/data/soma_field_median_iqr_by_field.csv'
SOMA_OBJECTS_CSV = r'/app/data/projects/globias_3d_nuclei_soma/processed_images/batch/batch_soma_objects_combined.csv'
FIG_DIR = r'/app/data/projects/globias_3d_nuclei_soma/figures'

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
})
sns.set_theme(style='whitegrid')
sns.set_palette('colorblind')
PALETTE = sns.color_palette('colorblind')
BLUE = PALETTE[0]   # #0173b2
ORANGE = PALETTE[1] # #de8f05
GREEN = PALETTE[2]
GRAY = '#4a4a4a'


def require_columns(df, required, df_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"VERIFICATION FAILED: {df_name} missing required columns: {missing}")


def verify_file(path):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise ValueError(f"VERIFICATION FAILED: output missing or empty -> {path}")


def save_dual(fig, stem):
    png_path = os.path.join(FIG_DIR, f'{stem}.png')
    svg_path = os.path.join(FIG_DIR, f'{stem}.svg')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    verify_file(png_path)
    verify_file(svg_path)
    return png_path, svg_path


def field_sort_key(field_id):
    s = str(field_id)
    if len(s) > 1 and s[0].upper() == 'F':
        num = s[1:]
        if num.isdigit():
            return int(num)
    return s


def sorted_field_ids(series):
    unique_ids = list(series.astype(str).unique())
    return sorted(unique_ids, key=field_sort_key)


def _normalize_filter_values(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def find_stats_row(stats_df, required_terms, analysis_category=None, dataset_subset=None, test_name_contains=None):
    require_columns(
        stats_df,
        ['analysis_id', 'analysis_category', 'dataset_subset', 'variable', 'test_name', 'estimate', 'ci_lower', 'ci_upper', 'p_value', 'n_fields'],
        'Statistics_Results.csv'
    )

    category_values = _normalize_filter_values(analysis_category)
    subset_values = _normalize_filter_values(dataset_subset)

    mask = pd.Series(True, index=stats_df.index)
    if category_values is not None:
        mask &= stats_df['analysis_category'].astype(str).str.lower().isin(category_values)
    if subset_values is not None:
        mask &= stats_df['dataset_subset'].astype(str).str.lower().isin(subset_values)
    if test_name_contains is not None:
        mask &= stats_df['test_name'].astype(str).str.lower().str.contains(str(test_name_contains).lower(), regex=False)

    search_blob = (
        stats_df['analysis_id'].astype(str).str.lower() + ' ' +
        stats_df['variable'].astype(str).str.lower() + ' ' +
        stats_df['test_name'].astype(str).str.lower()
    )
    for term in required_terms:
        mask &= search_blob.str.contains(str(term).lower(), regex=False)

    candidates = stats_df.loc[mask]
    if len(candidates) == 0:
        available_ids = sorted(stats_df['analysis_id'].astype(str).unique().tolist())
        available_categories = sorted(stats_df['analysis_category'].astype(str).unique().tolist())
        available_subsets = sorted(stats_df['dataset_subset'].astype(str).unique().tolist())
        available_variables = sorted(stats_df['variable'].astype(str).unique().tolist())
        raise ValueError(
            "VERIFICATION FAILED: could not find stats row for "
            f"terms={required_terms}, analysis_category={analysis_category}, "
            f"dataset_subset={dataset_subset}, test_name_contains={test_name_contains}. "
            f"Available analysis_id values: {available_ids}. "
            f"Available analysis_category values: {available_categories}. "
            f"Available dataset_subset values: {available_subsets}. "
            f"Available variable values: {available_variables}."
        )
    if len(candidates) > 1:
        print(f"WARNING: multiple matching stats rows for terms {required_terms}; using the first by file order.")
    return candidates.iloc[0]


def find_stats_row_exact(stats_df, analysis_id, analysis_category, dataset_subset):
    require_columns(
        stats_df,
        ['analysis_id', 'analysis_category', 'dataset_subset', 'estimate', 'ci_lower', 'ci_upper', 'p_value', 'n_fields'],
        'Statistics_Results.csv'
    )

    aid = str(analysis_id).strip().lower()
    cat = str(analysis_category).strip().lower()
    subset = str(dataset_subset).strip().lower()

    mask = (
        stats_df['analysis_id'].astype(str).str.strip().str.lower().eq(aid)
        & stats_df['analysis_category'].astype(str).str.strip().str.lower().eq(cat)
        & stats_df['dataset_subset'].astype(str).str.strip().str.lower().eq(subset)
    )
    candidates = stats_df.loc[mask]

    if len(candidates) == 0:
        available = stats_df[['analysis_id', 'analysis_category', 'dataset_subset']].drop_duplicates()
        raise ValueError(
            'VERIFICATION FAILED: exact ratio row not found for '
            f"analysis_id='{analysis_id}', analysis_category='{analysis_category}', dataset_subset='{dataset_subset}'. "
            f"Available triplets: {available.to_dict(orient='records')}"
        )
    if len(candidates) > 1:
        raise ValueError(
            'VERIFICATION FAILED: expected exactly one stats row for '
            f"analysis_id='{analysis_id}', analysis_category='{analysis_category}', dataset_subset='{dataset_subset}', "
            f'found {len(candidates)} rows.'
        )

    return candidates.iloc[0]


def iqr_outlier_count(series):
    vals = series.dropna().astype(float)
    if len(vals) == 0:
        return 0
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(((vals < low) | (vals > high)).sum())


def plot_counts_panels(counts_df, field_order):
    x = np.arange(len(field_order))
    plot_df = counts_df.set_index('field_id').loc[field_order].reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'hspace': 0.08})
    ax_top, ax_bottom = axes

    ax_top.bar(x, plot_df['dapi_total_nuclei_count'].values, color=BLUE, alpha=0.9, width=0.75, label='DAPI nuclei')
    ax_top.plot(x, plot_df['dapi_total_nuclei_count'].values, color=GRAY, marker='o', linewidth=1.2, markersize=4)
    ax_top.set_ylabel('Nuclei count per field (objects)')
    ax_top.set_title('Per-field 3D object counts from micro_sam segmentation (n=20 fields)', pad=10)
    ax_top.legend(frameon=False, loc='upper right')

    ax_bottom.bar(x, plot_df['soma_total_object_count'].values, color=ORANGE, alpha=0.9, width=0.75, label='Soma-marker objects')
    ax_bottom.plot(x, plot_df['soma_total_object_count'].values, color=GRAY, marker='s', linewidth=1.2, markersize=4)
    ax_bottom.set_ylabel('Soma object count per field (objects)')
    ax_bottom.set_xlabel('Field ID')
    ax_bottom.legend(frameon=False, loc='upper right')

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(field_order, rotation=45, ha='right')

    fig.text(
        0.01,
        0.01,
        '3D micro_sam AIS segmentation; paired DAPI and soma-marker fields. Separate y-scales retained for readability.',
        fontsize=11,
        ha='left'
    )

    return save_dual(fig, 'figure1_per_field_counts_panels')


def plot_ratio_panels(counts_df, stats_df, field_order):
    x = np.arange(len(field_order))
    plot_df = counts_df.set_index('field_id').loc[field_order].reset_index()

    total_row = find_stats_row_exact(
        stats_df,
        analysis_id='ratio_nuclei_to_soma_ratio_total',
        analysis_category='ratio_summary',
        dataset_subset='all_objects'
    )

    nonborder_row = find_stats_row_exact(
        stats_df,
        analysis_id='ratio_nuclei_to_soma_ratio_nonborder',
        analysis_category='ratio_summary',
        dataset_subset='nonborder_counts'
    )

    for row_name, row in [('total ratio', total_row), ('nonborder ratio', nonborder_row)]:
        est = float(row['estimate'])
        lo = float(row['ci_lower'])
        hi = float(row['ci_upper'])
        if lo > hi:
            raise ValueError(f"VERIFICATION FAILED: {row_name} CI lower > upper")
        if est < 0:
            print(f"WARNING: {row_name} median is negative ({est}); check upstream semantics.")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'hspace': 0.10})
    ax1, ax2 = axes

    y_total = plot_df['nuclei_to_soma_ratio_total_safe'].values
    ax1.plot(x, y_total, marker='o', linestyle='-', color=BLUE, linewidth=1.8, markersize=5, label='Per-field total ratio')
    ax1.axhline(float(total_row['estimate']), color=BLUE, linestyle='--', linewidth=1.6, label='Median (Statistics_Results.csv)')
    ax1.fill_between(x, float(total_row['ci_lower']), float(total_row['ci_upper']), color=BLUE, alpha=0.18, label='Bootstrap 95% CI')
    ax1.set_ylabel('Nuclei/Soma ratio (total counts)')
    ax1.set_title('Per-field nuclei-to-soma ratios with precomputed bootstrap median CI', pad=10)
    ax1.legend(frameon=False, loc='upper right')

    y_non = plot_df['nuclei_to_soma_ratio_nonborder_safe'].values
    ax2.plot(x, y_non, marker='s', linestyle='-', color=GREEN, linewidth=1.8, markersize=5, label='Per-field nonborder ratio')
    ax2.axhline(float(nonborder_row['estimate']), color=GREEN, linestyle='--', linewidth=1.6, label='Median (Statistics_Results.csv)')
    ax2.fill_between(x, float(nonborder_row['ci_lower']), float(nonborder_row['ci_upper']), color=GREEN, alpha=0.18, label='Bootstrap 95% CI')
    ax2.set_ylabel('Nuclei/Soma ratio (nonborder counts)')
    ax2.set_xlabel('Field ID')
    ax2.legend(frameon=False, loc='upper right')

    ax2.set_xticks(x)
    ax2.set_xticklabels(field_order, rotation=45, ha='right')

    fig.text(
        0.01,
        0.01,
        'Ratios plotted per field; horizontal lines and CI bands are read from Statistics_Results.csv (no recomputation).',
        fontsize=11,
        ha='left'
    )

    return save_dual(fig, 'figure2_per_field_ratios_with_bootstrap_ci')


def plot_scatter_with_annotation(counts_df, stats_df):
    require_columns(counts_df, ['field_id', 'dapi_total_nuclei_count', 'soma_total_object_count'], 'field_counts_with_safe_ratios.csv')

    pearson_row = find_stats_row(
        stats_df,
        required_terms=['total', 'counts', 'pearson'],
        analysis_category='association',
        dataset_subset='total_counts'
    )
    spearman_row = find_stats_row(
        stats_df,
        required_terms=['total', 'counts', 'spearman'],
        analysis_category='association',
        dataset_subset='total_counts'
    )

    r = float(pearson_row['estimate'])
    ci_low = float(pearson_row['ci_lower'])
    ci_high = float(pearson_row['ci_upper'])
    p = float(pearson_row['p_value'])
    n_fields = int(pearson_row['n_fields'])
    rho = float(spearman_row['estimate'])
    p_s = float(spearman_row['p_value'])

    if not (-1.0000001 <= r <= 1.0000001 and -1.0000001 <= rho <= 1.0000001):
        raise ValueError('VERIFICATION FAILED: correlation coefficient outside [-1, 1].')
    if not (0.0 <= p <= 1.0 and 0.0 <= p_s <= 1.0):
        raise ValueError('VERIFICATION FAILED: p-value outside [0, 1].')
    if ci_low > ci_high:
        raise ValueError('VERIFICATION FAILED: Pearson CI lower is greater than CI upper.')

    fig, ax = plt.subplots(figsize=(9, 7.2))

    sns.regplot(
        data=counts_df,
        x='dapi_total_nuclei_count',
        y='soma_total_object_count',
        ax=ax,
        ci=95,
        scatter_kws={'s': 52, 'alpha': 0.80, 'edgecolors': 'white', 'linewidths': 0.6, 'color': BLUE, 'marker': 'o'},
        line_kws={'color': ORANGE, 'linewidth': 2.0}
    )

    for _, row in counts_df.iterrows():
        ax.text(
            float(row['dapi_total_nuclei_count']) + 25,
            float(row['soma_total_object_count']) + 2,
            str(row['field_id']),
            fontsize=9,
            color=GRAY,
            alpha=0.9
        )

    significance_text = 'not significant at α=0.05' if p >= 0.05 else 'significant at α=0.05'
    ann = (
        f"Pearson r = {r:.6f}, 95% CI [{ci_low:.6f}, {ci_high:.6f}], p = {p:.6g}, n = {n_fields}\n"
        f"Spearman ρ = {rho:.6f}, p = {p_s:.7g}\n"
        f"Association is {significance_text}."
    )
    ax.text(
        0.03,
        0.97,
        ann,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=11.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.75', alpha=0.92)
    )

    ax.set_xlabel('DAPI nuclei count per field (objects)')
    ax.set_ylabel('Soma-marker object count per field (objects)')
    ax.set_title('Field-level association of nuclei and soma counts (3D micro_sam, n=20)')

    fig.text(
        0.01,
        0.01,
        'Descriptive linear fit shown for visualization only. Correlation statistics are read from Statistics_Results.csv.',
        fontsize=11,
        ha='left'
    )

    print(f"Scatter annotation values: Pearson r={r:.6f}, p={p:.6g}; Spearman rho={rho:.6f}, p={p_s:.7g}; n={n_fields}")

    return save_dual(fig, 'figure3_counts_scatter_association')


def plot_soma_distribution(soma_objects_df, soma_field_summary_df):
    require_columns(
        soma_objects_df,
        ['field_id', 'voxel_volume', 'touches_any_border'],
        'batch_soma_objects_combined.csv'
    )
    require_columns(
        soma_field_summary_df,
        ['field_id', 'n_soma_objects', 'soma_voxel_median', 'soma_voxel_q1', 'soma_voxel_q3', 'subset'],
        'soma_field_median_iqr_by_field.csv'
    )

    nonborder_df = soma_objects_df.loc[~soma_objects_df['touches_any_border']].copy()
    if nonborder_df.empty:
        raise ValueError('VERIFICATION FAILED: no nonborder soma objects available for the main voxel-volume distribution panel.')

    ratio = float(nonborder_df['voxel_volume'].max() / max(1.0, nonborder_df['voxel_volume'].min()))
    use_log = ratio >= 20.0

    summary_nb = soma_field_summary_df.loc[soma_field_summary_df['subset'].astype(str).str.lower() == 'nonborder_objects'].copy()
    if summary_nb.empty:
        print('WARNING: nonborder field summary subset not found; using all_objects for companion median/IQR panel.')
        summary_nb = soma_field_summary_df.loc[soma_field_summary_df['subset'].astype(str).str.lower() == 'all_objects'].copy()

    if summary_nb.empty:
        raise ValueError('VERIFICATION FAILED: neither nonborder_objects nor all_objects subsets found in soma_field_median_iqr_by_field.csv.')

    field_order = sorted_field_ids(summary_nb['field_id'])
    summary_nb['field_id'] = summary_nb['field_id'].astype(str)
    summary_nb = summary_nb.set_index('field_id').loc[field_order].reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={'width_ratios': [1.25, 1.0], 'wspace': 0.28})

    sns.histplot(
        nonborder_df,
        x='voxel_volume',
        bins=60,
        kde=True,
        color=ORANGE,
        edgecolor='white',
        linewidth=0.3,
        ax=ax1
    )
    if use_log:
        ax1.set_xscale('log')
    ax1.set_xlabel('Soma object volume (voxels)')
    ax1.set_ylabel('Object count (nonborder only)')
    ax1.set_title('Soma voxel-volume distribution\n(excluding objects touching any border)')

    y_pos = np.arange(len(summary_nb))
    med = summary_nb['soma_voxel_median'].astype(float).values
    err_low = (summary_nb['soma_voxel_median'] - summary_nb['soma_voxel_q1']).astype(float).values
    err_high = (summary_nb['soma_voxel_q3'] - summary_nb['soma_voxel_median']).astype(float).values

    ax2.errorbar(
        med,
        y_pos,
        xerr=[err_low, err_high],
        fmt='o',
        color=BLUE,
        ecolor=GRAY,
        elinewidth=1.4,
        capsize=2.8,
        markersize=4.8
    )
    if use_log:
        ax2.set_xscale('log')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(summary_nb['field_id'].tolist())
    ax2.set_xlabel('Field median soma volume (voxels) with IQR')
    ax2.set_ylabel('Field ID')
    ax2.set_title('Field-level medians ± IQR\n(fields remain identifiable)')

    fig.suptitle('Soma size summaries from 3D micro_sam segmentation (n=20 paired fields)', y=1.02)
    fig.text(
        0.01,
        -0.01,
        'Primary size unit is voxels. Physical μm³ omitted because calibration is provisional.',
        fontsize=11,
        ha='left'
    )

    print(
        f"Soma distribution panel: nonborder objects={len(nonborder_df)}, "
        f"fields in companion panel={summary_nb['field_id'].nunique()}, x-axis log-scale={'yes' if use_log else 'no'}"
    )

    return save_dual(fig, 'figure4_soma_voxel_distribution_and_field_iqr')


def main():
    stats_df = pd.read_csv(STATS_CSV)
    counts_df = pd.read_csv(COUNTS_CSV)
    soma_field_summary_df = pd.read_csv(FIELD_SOMA_SUMMARY_CSV)
    soma_objects_df = pd.read_csv(SOMA_OBJECTS_CSV)

    require_columns(
        counts_df,
        [
            'field_id',
            'dapi_total_nuclei_count',
            'soma_total_object_count',
            'nuclei_to_soma_ratio_total_safe',
            'nuclei_to_soma_ratio_nonborder_safe'
        ],
        'field_counts_with_safe_ratios.csv'
    )

    # Keep the intended biological replicate granularity (field-level; n=20).
    if 'field_status' in counts_df.columns:
        before = len(counts_df)
        counts_df = counts_df.loc[counts_df['field_status'].astype(str).str.lower() == 'ok'].copy()
        if len(counts_df) < before:
            print(f"WARNING: filtered to field_status=='ok' ({len(counts_df)}/{before} rows retained).")

    if counts_df.empty:
        raise ValueError('VERIFICATION FAILED: no rows available in field_counts_with_safe_ratios.csv after filtering.')

    field_order = sorted_field_ids(counts_df['field_id'])
    if len(field_order) != counts_df['field_id'].nunique():
        raise ValueError('VERIFICATION FAILED: field ordering inconsistency.')

    # Report IQR outliers (diagnostic only; no removal in plotting stage).
    print('IQR outlier counts (k=1.5, reported only):')
    print(f"  dapi_total_nuclei_count: {iqr_outlier_count(counts_df['dapi_total_nuclei_count'])}")
    print(f"  soma_total_object_count: {iqr_outlier_count(counts_df['soma_total_object_count'])}")
    print(f"  nuclei_to_soma_ratio_total_safe: {iqr_outlier_count(counts_df['nuclei_to_soma_ratio_total_safe'])}")
    nb_objects = soma_objects_df.loc[~soma_objects_df['touches_any_border'], 'voxel_volume']
    print(f"  soma voxel_volume (nonborder objects): {iqr_outlier_count(nb_objects)}")

    out_paths = []
    out_paths.extend(plot_counts_panels(counts_df, field_order))
    out_paths.extend(plot_ratio_panels(counts_df, stats_df, field_order))
    out_paths.extend(plot_scatter_with_annotation(counts_df, stats_df))
    out_paths.extend(plot_soma_distribution(soma_objects_df, soma_field_summary_df))

    if len(out_paths) != 8:
        raise ValueError(f"VERIFICATION FAILED: expected 8 output files (4 figures x PNG/SVG), got {len(out_paths)}")

    print('Saved figure files:')
    for p in out_paths:
        print(p)


if __name__ == '__main__':
    main()
