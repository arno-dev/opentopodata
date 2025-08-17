import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

from rasterio.enums import Resampling
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds

from opentopodata import utils

# Increased interpolation methods support
INTERPOLATION_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
}

# Thread-local storage for rasterio environments
_thread_local = threading.local()

class InputError(ValueError):
    """Invalid input data.

    The error message should be safe to pass back to the client.
    """


def _noop(x):
    return x


def _get_thread_local_env():
    """Get thread-local GDAL environment settings."""
    if not hasattr(_thread_local, 'env_configured'):
        # Configure GDAL environment per thread
        import os
        os.environ.setdefault('GDAL_CACHEMAX', '1024')
        os.environ.setdefault('GDAL_NUM_THREADS', 'ALL_CPUS')
        _thread_local.env_configured = True
    return _thread_local


@lru_cache(maxsize=128)
def _get_raster_metadata(path):
    """Cache raster metadata to avoid repeated file opens."""
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


def _validate_points_lie_within_raster(xs, ys, lats, lons, bounds, res):
    """Vectorized bounds checking for better performance."""
    # Convert to numpy arrays for vectorized operations
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    
    # Get actual extent with epsilon for floating point precision
    atol = 1e-8
    x_min = min(bounds.left, bounds.right) + abs(res[0]) / 2 - atol
    x_max = max(bounds.left, bounds.right) - abs(res[0]) / 2 + atol
    y_min = min(bounds.top, bounds.bottom) + abs(res[1]) / 2 - atol
    y_max = max(bounds.top, bounds.bottom) - abs(res[1]) / 2 + atol

    # Vectorized bounds checking
    x_in_bounds = (xs >= x_min) & (xs <= x_max)
    y_in_bounds = (ys >= y_min) & (ys <= y_max)
    
    # Find out of bounds indices
    oob_mask = ~(x_in_bounds & y_in_bounds)
    oob_indices = np.where(oob_mask)[0].tolist()
    
    return oob_indices


def _get_elevation_from_path_batch(lats, lons, path, interpolation):
    """Optimized batch elevation reading with windowed access."""
    _get_thread_local_env()  # Ensure thread-local configuration
    
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    
    try:
        with rasterio.open(path) as f:
            if f.crs is None:
                msg = "Dataset has no coordinate reference system."
                msg += f" Check the file '{path}' is a geo raster."
                raise InputError(msg)

            # Reproject coordinates
            try:
                if f.crs.is_epsg_code:
                    xs, ys = utils.reproject_latlons(lats, lons, epsg=f.crs.to_epsg())
                else:
                    xs, ys = utils.reproject_latlons(lats, lons, wkt=f.crs.to_wkt())
            except ValueError:
                raise InputError("Unable to transform latlons to dataset projection.")

            # Check bounds
            oob_indices = _validate_points_lie_within_raster(
                xs, ys, lats, lons, f.bounds, f.res
            )
            
            # Convert to numpy arrays for vectorized operations
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            
            # Get row/col indices
            rows, cols = f.index(xs, ys, op=_noop)
            rows = np.atleast_1d(rows) - 0.5
            cols = np.atleast_1d(cols) - 0.5
            
            # Clip to valid bounds
            rows = np.clip(rows, 0, f.height - 1)
            cols = np.clip(cols, 0, f.width - 1)
            
            # Group nearby points for efficient windowed reading
            z_all = _read_elevations_windowed(f, rows, cols, oob_indices, interpolation)
            
    except rasterio.RasterioIOError as e:
        if "not recognized as a supported file format" in str(e):
            msg = f"Dataset file '{path}' not recognised as a geo raster."
            raise InputError(msg)
        raise e

    return z_all


def _read_elevations_windowed(raster_file, rows, cols, oob_indices, interpolation):
    """Read elevations using optimized windowing strategy."""
    z_all = []
    interpolation_method = INTERPOLATION_METHODS.get(interpolation)
    oob_set = set(oob_indices)
    
    # For large batches, try to read larger windows to reduce I/O
    if len(rows) > 50:
        return _read_elevations_large_batch(raster_file, rows, cols, oob_set, interpolation_method)
    
    # For smaller batches, use individual reads
    for i, (row, col) in enumerate(zip(rows, cols)):
        if i in oob_set:
            z_all.append(None)
            continue
            
        window = Window(col, row, 1, 1)
        try:
            z_array = raster_file.read(
                indexes=1,
                window=window,
                resampling=interpolation_method,
                out_dtype=float,
                boundless=True,
                masked=True,
            )
            z = np.ma.filled(z_array, np.nan)[0][0]
            z_all.append(z)
        except Exception:
            z_all.append(None)
    
    return z_all


def _read_elevations_large_batch(raster_file, rows, cols, oob_set, interpolation_method):
    """Optimized reading for large batches using spatial clustering."""
    z_all = [None] * len(rows)
    
    # Group points by spatial proximity to minimize window reads
    valid_indices = [i for i in range(len(rows)) if i not in oob_set]
    
    if not valid_indices:
        return z_all
    
    # Calculate bounding box for all valid points
    valid_rows = rows[valid_indices]
    valid_cols = cols[valid_indices]
    
    min_row, max_row = int(np.floor(valid_rows.min())), int(np.ceil(valid_rows.max()))
    min_col, max_col = int(np.floor(valid_cols.min())), int(np.ceil(valid_cols.max()))
    
    # If the bounding box is reasonable, read the entire region
    window_height = max_row - min_row + 1
    window_width = max_col - min_col + 1
    
    # Only use large window if it's efficient (not too much extra data)
    efficiency_threshold = 4  # Read at most 4x the needed pixels
    needed_pixels = len(valid_indices)
    window_pixels = window_height * window_width
    
    if window_pixels <= needed_pixels * efficiency_threshold and window_pixels < 10000:
        # Read large window
        window = Window(min_col, min_row, window_width, window_height)
        try:
            data = raster_file.read(
                indexes=1,
                window=window,
                resampling=interpolation_method,
                out_dtype=float,
                boundless=True,
                masked=True,
            )
            data = np.ma.filled(data, np.nan)
            
            # Extract values for each point
            for i in valid_indices:
                local_row = int(rows[i] - min_row)
                local_col = int(cols[i] - min_col)
                if 0 <= local_row < window_height and 0 <= local_col < window_width:
                    z_all[i] = data[local_row, local_col]
                    
        except Exception:
            # Fall back to individual reads
            return _read_individual_points(raster_file, rows, cols, oob_set, interpolation_method)
    else:
        # Use individual reads for scattered points
        return _read_individual_points(raster_file, rows, cols, oob_set, interpolation_method)
    
    return z_all


def _read_individual_points(raster_file, rows, cols, oob_set, interpolation_method):
    """Fall back to individual point reading."""
    z_all = []
    
    for i, (row, col) in enumerate(zip(rows, cols)):
        if i in oob_set:
            z_all.append(None)
            continue
            
        window = Window(col, row, 1, 1)
        try:
            z_array = raster_file.read(
                indexes=1,
                window=window,
                resampling=interpolation_method,
                out_dtype=float,
                boundless=True,
                masked=True,
            )
            z = np.ma.filled(z_array, np.nan)[0][0]
            z_all.append(z)
        except Exception:
            z_all.append(None)
    
    return z_all


def _get_elevation_for_single_dataset(
    lats, lons, dataset, interpolation="nearest", nodata_value=None, max_workers=4
):
    """Parallel elevation reading for single dataset."""
    lats = np.array(lats)
    lons = np.array(lons)
    paths = dataset.location_paths(lats, lons)

    # Group by path for batch processing
    path_to_indices = collections.defaultdict(list)
    for i, path in enumerate(paths):
        path_to_indices[path].append(i)

    elevations = [None] * len(paths)
    
    # Use thread pool for parallel file processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {}
        
        for path, indices in path_to_indices.items():
            if path is None:
                for i in indices:
                    elevations[i] = None
                continue
                
            batch_lats = lats[indices]
            batch_lons = lons[indices]
            
            future = executor.submit(
                _get_elevation_from_path_batch, 
                batch_lats, batch_lons, path, interpolation
            )
            future_to_path[future] = (path, indices)
        
        # Collect results
        for future in as_completed(future_to_path):
            path, indices = future_to_path[future]
            try:
                path_elevations = future.result()
                for i, elevation in zip(indices, path_elevations):
                    elevations[i] = elevation
            except Exception as e:
                # Handle errors gracefully
                for i in indices:
                    elevations[i] = None

    elevations = utils.fill_na(elevations, nodata_value)
    return elevations


class _Point:
    def __init__(self, lat, lon, index):
        self.lat = lat
        self.lon = lon
        self.index = index
        self.elevation = None
        self.dataset_name = None


def get_elevation(lats, lons, datasets, interpolation="nearest", nodata_value=None):
    """Optimized elevation retrieval with parallel processing."""
    # Determine optimal number of workers based on CPU count and dataset size
    import os
    max_workers = min(len(datasets) * 2, os.cpu_count() or 4)
    
    points = [_Point(lat, lon, idx) for idx, (lat, lon) in enumerate(zip(lats, lons))]
    
    for dataset in datasets:
        # Filter points that need processing
        dataset_points = [p for p in points if p.elevation is None]
        if not dataset_points:
            break

        # Spatial filtering for dataset bounds
        dataset_points = [
            p for p in dataset_points 
            if (dataset.wgs84_bounds.bottom <= p.lat <= dataset.wgs84_bounds.top and
                dataset.wgs84_bounds.left <= p.lon <= dataset.wgs84_bounds.right)
        ]
        
        if not dataset_points:
            continue

        # Get elevations for this dataset
        elevations = _get_elevation_for_single_dataset(
            [p.lat for p in dataset_points],
            [p.lon for p in dataset_points],
            dataset,
            interpolation,
            nodata_value,
            max_workers=max_workers
        )

        # Update points with results
        for point, elevation in zip(dataset_points, elevations):
            if elevation is not None or points[point.index].elevation is None:
                points[point.index].elevation = elevation
                points[point.index].dataset_name = dataset.name

    # Return results
    fallback_dataset_name = datasets[-1].name if datasets else "unknown"
    dataset_names = [p.dataset_name or fallback_dataset_name for p in points]
    elevations = [p.elevation for p in points]
    
    return elevations, dataset_names