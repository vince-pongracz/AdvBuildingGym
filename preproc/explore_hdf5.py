"""
HDF5 File Explorer

Reads an HDF5 file and provides useful statistics, structure, and metadata.
Configuration is loaded from config.yaml in the same directory.

Usage:
    python -m preproc.explore_hdf5
    # or
    python preproc/explore_hdf5.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import h5py
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config.yaml in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = str(script_dir / "config.yaml")

    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path_obj}")

    with open(config_path_obj, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_dtype_info(dtype: np.dtype) -> dict:
    """Get human-readable dtype information."""
    return {
        "name": str(dtype),
        "kind": dtype.kind,  # 'f'=float, 'i'=int, 'u'=uint, 'S'=bytes, 'U'=unicode, etc.
        "itemsize": dtype.itemsize,
    }


def get_dataset_stats(dataset: h5py.Dataset, max_samples: int = 10) -> dict:
    """Compute statistics for a dataset."""
    stats = {
        "shape": dataset.shape,
        "dtype": get_dtype_info(dataset.dtype),
        "size": dataset.size,
        "nbytes": dataset.nbytes if hasattr(dataset, "nbytes") else dataset.size * dataset.dtype.itemsize,
        "chunks": dataset.chunks,
        "compression": dataset.compression,
        "compression_opts": dataset.compression_opts,
        "fillvalue": dataset.fillvalue,
    }

    # Add attributes
    if dataset.attrs:
        stats["attributes"] = {k: _convert_to_serializable(v) for k, v in dataset.attrs.items()}

    # Try to compute numerical statistics if applicable
    if dataset.size > 0 and dataset.dtype.kind in ("f", "i", "u"):
        try:
            data = dataset[:]
            stats["statistics"] = {
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "nan_count": int(np.isnan(data).sum()) if dataset.dtype.kind == "f" else 0,
                "inf_count": int(np.isinf(data).sum()) if dataset.dtype.kind == "f" else 0,
            }
        except Exception as e:
            stats["statistics"] = {"error": str(e)}

    # Sample values
    if dataset.size > 0:
        try:
            flat = dataset[:].flatten()
            sample_indices = np.linspace(0, len(flat) - 1, min(max_samples, len(flat)), dtype=int)
            stats["sample_values"] = [_convert_to_serializable(flat[i]) for i in sample_indices]
        except Exception as e:
            stats["sample_values"] = {"error": str(e)}

    return stats


def _convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    return obj


def explore_group(group: h5py.Group, max_samples: int = 10, indent: int = 0) -> dict:
    """Recursively explore an HDF5 group."""
    result = {
        "type": "group",
        "name": group.name,
        "num_items": len(group),
    }

    # Add attributes
    if group.attrs:
        result["attributes"] = {k: _convert_to_serializable(v) for k, v in group.attrs.items()}

    # Explore children
    children = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            children[key] = {
                "type": "dataset",
                **get_dataset_stats(item, max_samples),
            }
        elif isinstance(item, h5py.Group):
            children[key] = explore_group(item, max_samples, indent + 1)

    result["children"] = children
    return result


def print_structure(structure: dict, indent: int = 0) -> None:
    """Print the HDF5 structure in a readable format."""
    prefix = "  " * indent

    if structure["type"] == "group":
        print(f"{prefix}[Group] {structure['name']} ({structure['num_items']} items)")
        if "attributes" in structure and structure["attributes"]:
            print(f"{prefix}  Attributes: {list(structure['attributes'].keys())}")

        for name, child in structure.get("children", {}).items():
            print_structure(child, indent + 1)

    elif structure["type"] == "dataset":
        shape_str = "x".join(map(str, structure["shape"]))
        dtype_str = structure["dtype"]["name"]
        size_mb = structure["nbytes"] / (1024 * 1024)

        print(f"{prefix}[Dataset] shape=({shape_str}), dtype={dtype_str}, size={size_mb:.2f} MB")

        if "attributes" in structure and structure["attributes"]:
            print(f"{prefix}  Attributes: {list(structure['attributes'].keys())}")

        if structure.get("compression"):
            print(f"{prefix}  Compression: {structure['compression']} (opts={structure['compression_opts']})")

        if structure.get("chunks"):
            print(f"{prefix}  Chunks: {structure['chunks']}")

        if "statistics" in structure and isinstance(structure["statistics"], dict) and "error" not in structure["statistics"]:
            stats = structure["statistics"]
            print(f"{prefix}  Stats: min={stats['min']:.4g}, max={stats['max']:.4g}, "
                  f"mean={stats['mean']:.4g}, std={stats['std']:.4g}")
            if stats.get("nan_count", 0) > 0:
                print(f"{prefix}  NaN count: {stats['nan_count']}")
            if stats.get("inf_count", 0) > 0:
                print(f"{prefix}  Inf count: {stats['inf_count']}")

        if "sample_values" in structure and isinstance(structure["sample_values"], list):
            samples = structure["sample_values"][:5]  # Show first 5 for console
            print(f"{prefix}  Sample values: {samples}{'...' if len(structure['sample_values']) > 5 else ''}")


def explore_hdf5_file(file_path: str, config: dict) -> dict:
    """Explore an HDF5 file and return its structure."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    max_samples = config.get("max_sample_values", 10)
    target_groups = config.get("target_groups", [])

    with h5py.File(file_path, "r") as f:
        # File-level info
        result = {
            "file_path": str(file_path.absolute()),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "hdf5_version": f.libver,
        }

        # File-level attributes
        if f.attrs:
            result["file_attributes"] = {k: _convert_to_serializable(v) for k, v in f.attrs.items()}

        # Explore structure
        if target_groups:
            # Explore only specified groups
            result["structure"] = {}
            for group_path in target_groups:
                if group_path in f:
                    item = f[group_path]
                    if isinstance(item, h5py.Dataset):
                        result["structure"][group_path] = {
                            "type": "dataset",
                            **get_dataset_stats(item, max_samples),
                        }
                    elif isinstance(item, h5py.Group):
                        result["structure"][group_path] = explore_group(item, max_samples)
                else:
                    result["structure"][group_path] = {"error": f"Path not found: {group_path}"}
        else:
            # Explore entire file
            result["structure"] = explore_group(f, max_samples)

    return result


def main(config_path: str | None = None) -> None:
    """Main entry point."""
    # Load config
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    input_file = config.get("input_file")
    if not input_file:
        logger.error("No input_file specified in config.yaml")
        sys.exit(1)

    # Resolve relative paths from project root
    if not os.path.isabs(input_file):
        # Try relative to current working directory first
        if not os.path.exists(input_file):
            # Try relative to script directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            input_file = project_root / input_file

    logger.info(f"Exploring HDF5 file: {input_file}")

    try:
        result = explore_hdf5_file(str(input_file), config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading HDF5 file: {e}")
        sys.exit(1)

    output_format = config.get("output_format", "console")

    # Console output
    if output_format in ("console", "both"):
        print("\n" + "=" * 60)
        print("HDF5 FILE EXPLORATION REPORT")
        print("=" * 60)
        print(f"\nFile: {result['file_path']}")
        print(f"Size: {result['file_size_mb']:.2f} MB")
        print(f"HDF5 lib version: {result['hdf5_version']}")

        if "file_attributes" in result:
            print(f"\nFile attributes: {list(result['file_attributes'].keys())}")

        print("\n" + "-" * 60)
        print("STRUCTURE:")
        print("-" * 60)

        if isinstance(result["structure"], dict) and "type" in result["structure"]:
            print_structure(result["structure"])
        else:
            # Multiple target groups
            for path, struct in result["structure"].items():
                print(f"\n>>> {path}")
                if "error" in struct:
                    print(f"  Error: {struct['error']}")
                else:
                    print_structure(struct, indent=1)

        print("\n" + "=" * 60)

    # JSON output
    if output_format in ("json", "both"):
        output_json_path = config.get("output_json_path", "hdf5_report.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"JSON report saved to: {output_json_path}")


if __name__ == "__main__":
    # Allow passing config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
