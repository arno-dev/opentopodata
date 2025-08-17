import collections
import os
from functools import lru_cache

from rasterio.enums import Resampling
import numpy as np
import rasterio
from rasterio.windows import Window

from opentopodata import utils

INTERPOLATION_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}

class InputError(ValueError):
    """Invalid input data."""


def _noop(x):
    return x


# Configure GDAL for maximum I/O performance
def _configure_gdal_for_speed():
    """Configure GDAL environment for maximum speed."""
    os.environ.setdefault('GDAL_CACHEMAX', '512')  # 512MB cache
    os.environ.setdefault('GDAL_NUM_THREADS', 'ALL_CPUS')
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    os.environ.setdefault('VSI_CACHE', 'TRUE')
    os.environ.setdefault('VSI_CACHE_SIZE', '25000000')  # 25MB per file


@lru_cache(maxsize=64)
def _get_file_info_cached(path):
    """Cache file metadata to avoid repeated opens."""
    try:
        with rasterio.open(path) as f:
            return {
                'crs': f.crs,
                'bounds': f.bounds,
                'res': f.res,
                'height': f.height,
                'width': f.width,
                'transform': f.transform
            }
    except Exception:
        return None


def _validate_points_vectorized(xs, ys, bounds, res):
    """Vectorized bounds checking for speed."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    atol = 1e-8
    x_min = min(bounds.left, bounds.right) + abs(res[0]) / 2 - atol
    x_max = max(bounds.left, bounds.right) - abs(res[0]) / 2 + atol
    y_min = min(bounds.top, bounds.bottom) + abs(res[1]) / 2 - atol
    y_max = max(bounds.top, bounds.bottom) - abs(res[1]) / 2 + atol

    valid_mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    return set(np.where(~valid_mask)[0])


def _get_elevation_from_path_io_optimized(lats, lons, path, interpolation):
    """I/O optimized elevation reading - minimize file operations."""
    
    _configure_gdal_for_speed()
    
    # Use cached file info when possible
    file_info = _get_file_info_cached(path)
    if not file_info:
        return [None] * len(lats)
    
    interpolation_method = INTERPOLATION_METHODS.get(interpolation, Resampling.nearest)
    
    # Convert to numpy arrays for speed
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    
    try:
        with rasterio.open(path) as f:
            if f.crs is None:
                raise InputError("Dataset has no coordinate reference system.")

            # Fast coordinate transformation
            try:
                if f.crs.is_epsg_code:
                    xs, ys = utils.reproject_latlons(lats, lons, epsg=f.crs.to_epsg())
                else:
                    xs, ys = utils.reproject_latlons(lats, lons, wkt=f.crs.to_wkt())
            except ValueError:
                raise InputError("Unable to transform latlons to dataset projection.")

            # Fast vectorized bounds checking
            oob_indices = _validate_points_vectorized(xs, ys, f.bounds, f.res)
            
            # Vectorized coordinate processing
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            rows, cols = f.index(xs, ys, op=_noop)
            rows = np.atleast_1d(rows) - 0.5
            cols = np.atleast_1d(cols) - 0.5
            rows = np.clip(rows, 0, f.height - 1)
            cols = np.clip(cols, 0, f.width - 1)

            # KEY OPTIMIZATION: Smart batch reading strategy
            return _read_with_smart_batching(f, rows, cols, oob_indices, interpolation_method)

    except rasterio.RasterioIOError as e:
        if "not recognized as a supported file format" in str(e):
            raise InputError(f"Dataset file '{path}' not recognised as a geo raster.")
        raise e


def _read_with_smart_batching(f, rows, cols, oob_indices, interpolation_method):
    """Smart batching strategy to minimize I/O operations."""
    
    z_all = [None] * len(rows)
    oob_set = set(oob_indices)
    
    # Convert to integer pixel coordinates for grouping
    pixel_groups = collections.defaultdict(list)
    
    for i, (row, col) in enumerate(zip(rows, cols)):
        if i in oob_set:
            continue
        
        # Group by integer pixel coordinates
        pixel_row = int(np.floor(row))
        pixel_col = int(np.floor(col))
        pixel_groups[(pixel_row, pixel_col)].append((i, row, col))
    
    if not pixel_groups:
        return z_all
    
    # Strategy 1: If points are scattered, read larger regions
    unique_pixels = len(pixel_groups)
    total_points = len([i for i in range(len(rows)) if i not in oob_set])
    
    if unique_pixels > 50 and total_points > 50:
        # Many scattered points - use region-based reading
        return _read_scattered_points_optimized(f, pixel_groups, z_all, interpolation_method)
    else:
        # Fewer points - use enhanced individual reading
        return _read_clustered_points_optimized(f, pixel_groups, z_all, interpolation_method)


def _read_scattered_points_optimized(f, pixel_groups, z_all, interpolation_method):
    """Optimized reading for scattered points using larger windows."""
    
    # Find bounding box of all points
    all_pixels = list(pixel_groups.keys())
    min_row = min(p[0] for p in all_pixels)
    max_row = max(p[0] for p in all_pixels)
    min_col = min(p[1] for p in all_pixels)
    max_col = max(p[1] for p in all_pixels)
    
    # Calculate region dimensions
    region_height = max_row - min_row + 1
    region_width = max_col - min_col + 1
    total_pixels = region_height * region_width
    needed_pixels = len(all_pixels)
    
    # Only use region reading if efficient (not reading too much extra data)
    efficiency_ratio = total_pixels / needed_pixels
    
    if efficiency_ratio <= 4 and total_pixels <= 10000:  # Read at most 4x needed pixels
        # Read entire region at once
        window = Window(min_col, min_row, region_width, region_height)
        
        try:
            data = f.read(
                indexes=1,
                window=window,
                resampling=interpolation_method,
                out_dtype=np.float64,
                boundless=True,
                masked=True,
            )
            data = np.ma.filled(data, np.nan)
            
            # Extract values for each pixel group
            for (pixel_row, pixel_col), point_list in pixel_groups.items():
                local_row = pixel_row - min_row
                local_col = pixel_col - min_col
                
                if 0 <= local_row < region_height and 0 <= local_col < region_width:
                    pixel_value = data[local_row, local_col]
                    for i, row, col in point_list:
                        z_all[i] = pixel_value
                        
            return z_all
            
        except Exception:
            # Fall back to individual reads
            pass
    
    # Fall back to individual pixel reading
    return _read_clustered_points_optimized(f, pixel_groups, z_all, interpolation_method)


def _read_clustered_points_optimized(f, pixel_groups, z_all, interpolation_method):
    """Optimized reading for clustered points - read each unique pixel once."""
    
    # Read each unique pixel only once
    pixel_cache = {}
    
    for (pixel_row, pixel_col), point_list in pixel_groups.items():
        # Use first point's exact coordinates for window (maintains compatibility)
        _, first_row, first_col = point_list[0]
        
        window = Window(first_col, first_row, 1, 1)
        
        try:
            z_array = f.read(
                indexes=1,
                window=window,
                resampling=interpolation_method,
                out_dtype=np.float64,
                boundless=True,
                masked=True,
            )
            pixel_value = np.ma.filled(z_array, np.nan)[0][0]
            pixel_cache[(pixel_row, pixel_col)] = pixel_value
        except Exception:
            pixel_cache[(pixel_row, pixel_col)] = None
    
    # Assign cached values to all points in each pixel group
    for (pixel_row, pixel_col), point_list in pixel_groups.items():
        cached_value = pixel_cache.get((pixel_row, pixel_col))
        for i, row, col in point_list:
            z_all[i] = cached_value
    
    return z_all


def _get_elevation_for_single_dataset_io_optimized(
    lats, lons, dataset, interpolation="nearest", nodata_value=None
):
    """I/O optimized single dataset processing."""
    
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    paths = dataset.location_paths(lats, lons)

    # Group by path for batch processing
    path_to_indices = collections.defaultdict(list)
    for i, path in enumerate(paths):
        path_to_indices[path].append(i)

    elevations = [None] * len(paths)
    
    # Process each file with I/O optimizations
    for path, indices in path_to_indices.items():
        if path is None:
            for i in indices:
                elevations[i] = None
            continue
            
        batch_lats = lats[indices]
        batch_lons = lons[indices]
        
        path_elevations = _get_elevation_from_path_io_optimized(
            batch_lats, batch_lons, path, interpolation
        )
        
        for i, elevation in zip(indices, path_elevations):
            elevations[i] = elevation

    elevations = utils.fill_na(elevations, nodata_value)
    return elevations


class _Point:
    def __init__(self, lat, lon, index):
        self.lat = lat
        self.lon = lon
        self.index = index
        self.elevation = None
        self.dataset_name = None


def get_elevation_reference_compatible(lats, lons, datasets, interpolation="nearest", nodata_value=None):
    """I/O optimized elevation retrieval."""
    
    # Single dataset optimization (most common case)
    if len(datasets) == 1:
        elevations = _get_elevation_for_single_dataset_io_optimized(
            lats, lons, datasets[0], interpolation, nodata_value
        )
        dataset_names = [datasets[0].name] * len(lats)
        return elevations, dataset_names

    # Multiple dataset processing
    points = [_Point(lat, lon, idx) for idx, (lat, lon) in enumerate(zip(lats, lons))]
    for dataset in datasets:
        dataset_points = [p for p in points if p.elevation is None]
        if not dataset_points:
            break

        # Vectorized bounds filtering
        if dataset_points:
            lats_array = np.array([p.lat for p in dataset_points])
            lons_array = np.array([p.lon for p in dataset_points])
            
            bounds = dataset.wgs84_bounds
            valid_mask = (
                (lats_array >= bounds.bottom) & (lats_array <= bounds.top) &
                (lons_array >= bounds.left) & (lons_array <= bounds.right)
            )
            
            filtered_points = [p for i, p in enumerate(dataset_points) if valid_mask[i]]
            
            if not filtered_points:
                continue

            elevations = _get_elevation_for_single_dataset_io_optimized(
                [p.lat for p in filtered_points],
                [p.lon for p in filtered_points],
                dataset,
                interpolation,
                nodata_value,
            )

            for point, elevation in zip(filtered_points, elevations):
                points[point.index].elevation = elevation
                points[point.index].dataset_name = dataset.name

    fallback_dataset_name = datasets[-1].name
    dataset_names = [p.dataset_name or fallback_dataset_name for p in points]
    elevations = [p.elevation for p in points]
    return elevations, dataset_names


def get_elevation(lats, lons, datasets, interpolation="nearest", nodata_value=None):
    """Main elevation function - I/O optimized."""
    return get_elevation_reference_compatible(lats, lons, datasets, interpolation, nodata_value)


# Cache management
def clear_elevation_caches():
    """Clear caches to free memory."""
    _get_file_info_cached.cache_clear()