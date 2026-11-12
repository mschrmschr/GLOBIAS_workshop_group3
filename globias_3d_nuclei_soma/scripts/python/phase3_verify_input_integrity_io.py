from pathlib import Path
import json
import re
import traceback
import tifffile
import xml.etree.ElementTree as ET

# =========================
# CONFIG
# =========================
DAPI_DIR = Path(r'/data/tasks/Globias_workshop/DAPI')
STAIN_DIR = Path(r'/data/tasks/Globias_workshop/stain')
OUTPUT_DATA_DIR = Path(r'/app/data/projects/globias_3d_nuclei_soma/data')
LOG_DIR = Path(r'/app/data/projects/globias_3d_nuclei_soma/logs')

MANIFEST_CSV = OUTPUT_DATA_DIR / 'input_integrity_manifest.csv'
SUMMARY_JSON = LOG_DIR / 'input_integrity_summary.json'
SUMMARY_TXT = LOG_DIR / 'input_integrity_summary.txt'

FIELD_PATTERN = re.compile(r'(F\d{4})', re.IGNORECASE)
VALID_EXTENSIONS = {'.tif', '.tiff'}


def _field_id_from_name(path_obj):
    m = FIELD_PATTERN.search(path_obj.name)
    return m.group(1).upper() if m else None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _parse_ome_physical_sizes(ome_xml):
    out = {
        'ome_physical_size_x': np.nan,
        'ome_physical_size_y': np.nan,
        'ome_physical_size_z': np.nan,
        'ome_physical_size_x_unit': '',
        'ome_physical_size_y_unit': '',
        'ome_physical_size_z_unit': ''
    }
    if not ome_xml:
        return out

    try:
        root = ET.fromstring(ome_xml)
        pixels = None
        for elem in root.iter():
            if elem.tag.endswith('Pixels'):
                pixels = elem
                break
        if pixels is None:
            return out

        out['ome_physical_size_x'] = _safe_float(pixels.attrib.get('PhysicalSizeX', np.nan))
        out['ome_physical_size_y'] = _safe_float(pixels.attrib.get('PhysicalSizeY', np.nan))
        out['ome_physical_size_z'] = _safe_float(pixels.attrib.get('PhysicalSizeZ', np.nan))
        out['ome_physical_size_x_unit'] = str(pixels.attrib.get('PhysicalSizeXUnit', '') or '')
        out['ome_physical_size_y_unit'] = str(pixels.attrib.get('PhysicalSizeYUnit', '') or '')
        out['ome_physical_size_z_unit'] = str(pixels.attrib.get('PhysicalSizeZUnit', '') or '')
    except Exception:
        pass

    return out


def inspect_tiff(path_obj):
    info = {
        'path': str(path_obj),
        'exists': False,
        'is_file': False,
        'size_bytes': np.nan,
        'readable': False,
        'error': '',
        'series_shape': '',
        'series_axes': '',
        'ndim': np.nan,
        'dtype': '',
        'z_slices': np.nan,
        'y_pixels': np.nan,
        'x_pixels': np.nan,
        'n_pages': np.nan,
        'x_resolution_raw': '',
        'y_resolution_raw': '',
        'resolution_unit_raw': '',
        'x_spacing_from_resolution': np.nan,
        'y_spacing_from_resolution': np.nan,
        'imagej_spacing': np.nan,
        'imagej_unit': '',
        'ome_physical_size_x': np.nan,
        'ome_physical_size_y': np.nan,
        'ome_physical_size_z': np.nan,
        'ome_physical_size_x_unit': '',
        'ome_physical_size_y_unit': '',
        'ome_physical_size_z_unit': ''
    }

    try:
        info['exists'] = path_obj.exists()
        info['is_file'] = path_obj.is_file()
        if not info['exists'] or not info['is_file']:
            info['error'] = 'Path missing or not a file'
            return info

        info['size_bytes'] = float(path_obj.stat().st_size)

        with tifffile.TiffFile(str(path_obj)) as tif:
            if len(tif.series) == 0:
                info['error'] = 'No TIFF series found'
                return info

            series = tif.series[0]
            shape = tuple(series.shape)
            axes = str(series.axes)

            info['series_shape'] = str(shape)
            info['series_axes'] = axes
            info['ndim'] = int(len(shape))
            info['dtype'] = str(series.dtype)
            info['n_pages'] = int(len(tif.pages))

            if len(shape) >= 2:
                info['y_pixels'] = int(shape[-2])
                info['x_pixels'] = int(shape[-1])
            if len(shape) >= 3:
                info['z_slices'] = int(shape[0])
            elif len(shape) == 2:
                info['z_slices'] = 1

            # Decode at least one page to verify readability without loading full stack.
            _ = tif.pages[0].asarray()
            if len(tif.pages) > 1:
                _ = tif.pages[-1].asarray()

            tags = tif.pages[0].tags
            xres = tags.get('XResolution')
            yres = tags.get('YResolution')
            runit = tags.get('ResolutionUnit')

            xres_val = xres.value if xres is not None else None
            yres_val = yres.value if yres is not None else None
            runit_val = runit.value if runit is not None else None

            info['x_resolution_raw'] = str(xres_val) if xres_val is not None else ''
            info['y_resolution_raw'] = str(yres_val) if yres_val is not None else ''
            info['resolution_unit_raw'] = str(runit_val) if runit_val is not None else ''

            def _spacing_from_resolution(res_value):
                if res_value is None:
                    return np.nan
                try:
                    if isinstance(res_value, tuple) and len(res_value) == 2:
                        num = float(res_value[0])
                        den = float(res_value[1])
                        if num > 0:
                            return den / num
                        return np.nan
                    v = float(res_value)
                    if v > 0:
                        return 1.0 / v
                except Exception:
                    return np.nan
                return np.nan

            info['x_spacing_from_resolution'] = _spacing_from_resolution(xres_val)
            info['y_spacing_from_resolution'] = _spacing_from_resolution(yres_val)

            ij_meta = tif.imagej_metadata or {}
            info['imagej_spacing'] = _safe_float(ij_meta.get('spacing', np.nan))
            info['imagej_unit'] = str(ij_meta.get('unit', '') or '')

            ome_meta = tif.ome_metadata
            ome_sizes = _parse_ome_physical_sizes(ome_meta)
            info.update(ome_sizes)

            info['readable'] = True

    except Exception as e:
        info['error'] = f'{type(e).__name__}: {e}'

    return info


def map_fields_to_paths(folder):
    paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])
    field_map = {}
    extras_without_id = []
    for p in paths:
        fid = _field_id_from_name(p)
        if fid is None:
            extras_without_id.append(str(p))
            continue
        field_map.setdefault(fid, []).append(str(p))
    return paths, field_map, extras_without_id


def one_or_empty(path_list):
    if len(path_list) == 1:
        return path_list[0]
    return ''


def main():
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    errors_global = []

    # Input directory checks.
    for d in [DAPI_DIR, STAIN_DIR]:
        if not d.exists() or not d.is_dir():
            raise ValueError(f'VERIFICATION FAILED: Missing required input directory: {d}')

    dapi_files, dapi_map, dapi_noid = map_fields_to_paths(DAPI_DIR)
    stain_files, stain_map, stain_noid = map_fields_to_paths(STAIN_DIR)

    if dapi_noid:
        errors_global.append(f'DAPI files without F#### ID: {len(dapi_noid)}')
    if stain_noid:
        errors_global.append(f'Stain files without F#### ID: {len(stain_noid)}')

    all_field_ids = sorted(set(dapi_map.keys()) | set(stain_map.keys()))

    manifest_rows = []

    for fid in all_field_ids:
        dapi_list = dapi_map.get(fid, [])
        stain_list = stain_map.get(fid, [])

        dapi_path = one_or_empty(dapi_list)
        stain_path = one_or_empty(stain_list)

        dapi_info = inspect_tiff(Path(dapi_path)) if dapi_path else None
        stain_info = inspect_tiff(Path(stain_path)) if stain_path else None

        row_errors = []
        pair_has_single_file_each = (len(dapi_list) == 1 and len(stain_list) == 1)

        if len(dapi_list) == 0:
            row_errors.append('Missing DAPI file')
        if len(stain_list) == 0:
            row_errors.append('Missing stain file')
        if len(dapi_list) > 1:
            row_errors.append(f'Multiple DAPI files for field ({len(dapi_list)})')
        if len(stain_list) > 1:
            row_errors.append(f'Multiple stain files for field ({len(stain_list)})')

        dapi_readable = bool(dapi_info and dapi_info['readable'])
        stain_readable = bool(stain_info and stain_info['readable'])

        if dapi_info and dapi_info['error']:
            row_errors.append(f"DAPI read error: {dapi_info['error']}")
        if stain_info and stain_info['error']:
            row_errors.append(f"Stain read error: {stain_info['error']}")

        dapi_shape = dapi_info['series_shape'] if dapi_info else ''
        stain_shape = stain_info['series_shape'] if stain_info else ''
        shapes_identical = bool(dapi_shape and stain_shape and dapi_shape == stain_shape)

        if dapi_info and stain_info and not shapes_identical:
            row_errors.append('DAPI and stain shape mismatch')

        dapi_ndim = int(dapi_info['ndim']) if dapi_info and not pd.isna(dapi_info['ndim']) else np.nan
        stain_ndim = int(stain_info['ndim']) if stain_info and not pd.isna(stain_info['ndim']) else np.nan

        if dapi_info and dapi_ndim < 3:
            row_errors.append('DAPI is not 3D (ndim < 3)')
        if stain_info and stain_ndim < 3:
            row_errors.append('Stain is not 3D (ndim < 3)')

        dapi_axes = dapi_info['series_axes'] if dapi_info else ''
        stain_axes = stain_info['series_axes'] if stain_info else ''
        axes_identical = bool(dapi_axes and stain_axes and dapi_axes == stain_axes)
        if dapi_info and stain_info and not axes_identical:
            row_errors.append('DAPI and stain axes mismatch')

        pair_valid = bool(
            pair_has_single_file_each
            and dapi_readable
            and stain_readable
            and shapes_identical
            and dapi_ndim >= 3
            and stain_ndim >= 3
        )

        manifest_rows.append({
            'field_id': fid,
            'dapi_path': dapi_path,
            'stain_path': stain_path,
            'dapi_file_count_for_field': len(dapi_list),
            'stain_file_count_for_field': len(stain_list),
            'pair_has_single_file_each': pair_has_single_file_each,
            'dapi_readable': dapi_readable,
            'stain_readable': stain_readable,
            'dapi_shape': dapi_shape,
            'stain_shape': stain_shape,
            'pair_shape_identical': shapes_identical,
            'dapi_axes': dapi_axes,
            'stain_axes': stain_axes,
            'pair_axes_identical': axes_identical,
            'dapi_ndim': dapi_ndim,
            'stain_ndim': stain_ndim,
            'dapi_z_slices': dapi_info['z_slices'] if dapi_info else np.nan,
            'stain_z_slices': stain_info['z_slices'] if stain_info else np.nan,
            'dapi_y_pixels': dapi_info['y_pixels'] if dapi_info else np.nan,
            'dapi_x_pixels': dapi_info['x_pixels'] if dapi_info else np.nan,
            'stain_y_pixels': stain_info['y_pixels'] if stain_info else np.nan,
            'stain_x_pixels': stain_info['x_pixels'] if stain_info else np.nan,
            'dapi_dtype': dapi_info['dtype'] if dapi_info else '',
            'stain_dtype': stain_info['dtype'] if stain_info else '',
            'dapi_size_bytes': dapi_info['size_bytes'] if dapi_info else np.nan,
            'stain_size_bytes': stain_info['size_bytes'] if stain_info else np.nan,
            'dapi_n_pages': dapi_info['n_pages'] if dapi_info else np.nan,
            'stain_n_pages': stain_info['n_pages'] if stain_info else np.nan,
            'dapi_x_resolution_raw': dapi_info['x_resolution_raw'] if dapi_info else '',
            'dapi_y_resolution_raw': dapi_info['y_resolution_raw'] if dapi_info else '',
            'dapi_resolution_unit_raw': dapi_info['resolution_unit_raw'] if dapi_info else '',
            'dapi_x_spacing_from_resolution': dapi_info['x_spacing_from_resolution'] if dapi_info else np.nan,
            'dapi_y_spacing_from_resolution': dapi_info['y_spacing_from_resolution'] if dapi_info else np.nan,
            'dapi_imagej_spacing': dapi_info['imagej_spacing'] if dapi_info else np.nan,
            'dapi_imagej_unit': dapi_info['imagej_unit'] if dapi_info else '',
            'dapi_ome_physical_size_x': dapi_info['ome_physical_size_x'] if dapi_info else np.nan,
            'dapi_ome_physical_size_y': dapi_info['ome_physical_size_y'] if dapi_info else np.nan,
            'dapi_ome_physical_size_z': dapi_info['ome_physical_size_z'] if dapi_info else np.nan,
            'dapi_ome_physical_size_x_unit': dapi_info['ome_physical_size_x_unit'] if dapi_info else '',
            'dapi_ome_physical_size_y_unit': dapi_info['ome_physical_size_y_unit'] if dapi_info else '',
            'dapi_ome_physical_size_z_unit': dapi_info['ome_physical_size_z_unit'] if dapi_info else '',
            'stain_x_resolution_raw': stain_info['x_resolution_raw'] if stain_info else '',
            'stain_y_resolution_raw': stain_info['y_resolution_raw'] if stain_info else '',
            'stain_resolution_unit_raw': stain_info['resolution_unit_raw'] if stain_info else '',
            'stain_x_spacing_from_resolution': stain_info['x_spacing_from_resolution'] if stain_info else np.nan,
            'stain_y_spacing_from_resolution': stain_info['y_spacing_from_resolution'] if stain_info else np.nan,
            'stain_imagej_spacing': stain_info['imagej_spacing'] if stain_info else np.nan,
            'stain_imagej_unit': stain_info['imagej_unit'] if stain_info else '',
            'stain_ome_physical_size_x': stain_info['ome_physical_size_x'] if stain_info else np.nan,
            'stain_ome_physical_size_y': stain_info['ome_physical_size_y'] if stain_info else np.nan,
            'stain_ome_physical_size_z': stain_info['ome_physical_size_z'] if stain_info else np.nan,
            'stain_ome_physical_size_x_unit': stain_info['ome_physical_size_x_unit'] if stain_info else '',
            'stain_ome_physical_size_y_unit': stain_info['ome_physical_size_y_unit'] if stain_info else '',
            'stain_ome_physical_size_z_unit': stain_info['ome_physical_size_z_unit'] if stain_info else '',
            'pair_valid_for_downstream_3d': pair_valid,
            'errors': ' | '.join(row_errors)
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df = manifest_df.sort_values('field_id').reset_index(drop=True)

    # Global integrity checks.
    n_dapi_tiff = len(dapi_files)
    n_stain_tiff = len(stain_files)
    n_fields_union = len(all_field_ids)
    n_pairs_valid = int(manifest_df['pair_valid_for_downstream_3d'].sum()) if len(manifest_df) else 0

    exactly_20_by_id = (n_fields_union == 20)
    exactly_20_dapi_files = (n_dapi_tiff == 20)
    exactly_20_stain_files = (n_stain_tiff == 20)
    all_pairs_valid = bool(len(manifest_df) > 0 and manifest_df['pair_valid_for_downstream_3d'].all())

    all_safe_for_downstream_3d = bool(
        exactly_20_by_id
        and exactly_20_dapi_files
        and exactly_20_stain_files
        and all_pairs_valid
    )

    if not exactly_20_by_id:
        errors_global.append(f'Expected 20 field IDs across channels, found {n_fields_union}')
    if not exactly_20_dapi_files:
        errors_global.append(f'Expected 20 DAPI TIFF files, found {n_dapi_tiff}')
    if not exactly_20_stain_files:
        errors_global.append(f'Expected 20 stain TIFF files, found {n_stain_tiff}')
    if not all_pairs_valid:
        errors_global.append('One or more field pairs failed 3D compatibility checks')

    summary = {
        'input_dirs': {
            'dapi_dir': str(DAPI_DIR),
            'stain_dir': str(STAIN_DIR)
        },
        'output_files': {
            'manifest_csv': str(MANIFEST_CSV),
            'summary_json': str(SUMMARY_JSON),
            'summary_txt': str(SUMMARY_TXT)
        },
        'counts': {
            'dapi_tiff_files': n_dapi_tiff,
            'stain_tiff_files': n_stain_tiff,
            'field_ids_union': n_fields_union,
            'rows_in_manifest': int(len(manifest_df)),
            'valid_pairs': n_pairs_valid
        },
        'requirements': {
            'expected_pairs': 20,
            'exactly_20_by_id': exactly_20_by_id,
            'exactly_20_dapi_files': exactly_20_dapi_files,
            'exactly_20_stain_files': exactly_20_stain_files,
            'all_pairs_valid': all_pairs_valid
        },
        'all_pairs_safe_for_downstream_3d_processing': all_safe_for_downstream_3d,
        'global_errors': errors_global,
        'files_without_field_id': {
            'dapi': dapi_noid,
            'stain': stain_noid
        }
    }

    manifest_df.to_csv(MANIFEST_CSV, index=False)
    with open(SUMMARY_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    lines = [
        'Input Integrity / IO Check Summary',
        f"DAPI TIFF files: {n_dapi_tiff}",
        f"Stain TIFF files: {n_stain_tiff}",
        f"Unique field IDs: {n_fields_union}",
        f"Valid 3D pairs: {n_pairs_valid}/{len(manifest_df)}",
        f"All pairs safe for downstream 3D processing: {all_safe_for_downstream_3d}",
    ]
    if errors_global:
        lines.append('Global issues:')
        lines.extend([f'- {e}' for e in errors_global])
    else:
        lines.append('Global issues: none')

    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # Fail loudly only for true script/output invariants.
    if not MANIFEST_CSV.exists() or MANIFEST_CSV.stat().st_size == 0:
        raise ValueError('VERIFICATION FAILED: Manifest CSV was not written or is empty.')
    if not SUMMARY_JSON.exists() or SUMMARY_JSON.stat().st_size == 0:
        raise ValueError('VERIFICATION FAILED: Summary JSON was not written or is empty.')
    if not SUMMARY_TXT.exists() or SUMMARY_TXT.stat().st_size == 0:
        raise ValueError('VERIFICATION FAILED: Summary TXT was not written or is empty.')

    required_manifest_columns = {
        'field_id', 'dapi_path', 'stain_path', 'pair_valid_for_downstream_3d', 'errors'
    }
    missing_cols = required_manifest_columns - set(manifest_df.columns)
    if missing_cols:
        raise ValueError(f'VERIFICATION FAILED: Manifest missing required columns: {sorted(missing_cols)}')

    print('IO integrity check complete.')
    print(f'Manifest: {MANIFEST_CSV}')
    print(f'Summary JSON: {SUMMARY_JSON}')
    print(f'Summary TXT: {SUMMARY_TXT}')
    print(f'Valid pairs: {n_pairs_valid}/{len(manifest_df)}')
    print(f'All pairs safe for downstream 3D processing: {all_safe_for_downstream_3d}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('FATAL ERROR during input integrity check:')
        print(traceback.format_exc())
        raise
