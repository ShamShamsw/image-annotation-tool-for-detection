"""
storage.py - Persistence layer for Project 36: Image Annotation Tool For Detection
"""

from pathlib import Path
import json
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent / 'data'
OUTPUTS_DIR = DATA_DIR / 'outputs'
RUN_CATALOG_FILE = DATA_DIR / 'runs.json'
ANNOTATION_CATALOG_FILE = DATA_DIR / 'annotations.json'
LIBRARY_FILE = DATA_DIR / 'library.json'


def ensure_data_dirs() -> None:
    """Create local data/output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data_dir() -> None:
    """Backwards-compatible alias for older scaffold code."""
    ensure_data_dirs()


def _load_list_file(path: Path) -> List[Dict[str, Any]]:
    """Safely load a list-based JSON file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_list_file(path: Path, data: List[Dict[str, Any]]) -> None:
    """Persist list-based JSON with indentation."""
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_run_catalog() -> List[Dict[str, Any]]:
    """Load run catalog entries from disk."""
    ensure_data_dirs()
    return _load_list_file(RUN_CATALOG_FILE)


def save_run_catalog(catalog: List[Dict[str, Any]]) -> None:
    """Persist run catalog to disk."""
    ensure_data_dirs()
    _save_list_file(RUN_CATALOG_FILE, catalog)


def load_annotation_catalog() -> List[Dict[str, Any]]:
    """Load annotation catalog entries from disk."""
    ensure_data_dirs()
    return _load_list_file(ANNOTATION_CATALOG_FILE)


def save_annotation_catalog(catalog: List[Dict[str, Any]]) -> None:
    """Persist annotation catalog to disk."""
    ensure_data_dirs()
    _save_list_file(ANNOTATION_CATALOG_FILE, catalog)


def load_library() -> List[Dict[str, Any]]:
    """Load image annotation library from disk."""
    ensure_data_dirs()
    return _load_list_file(LIBRARY_FILE)


def save_library(library: List[Dict[str, Any]]) -> str:
    """Persist image annotation library to disk."""
    ensure_data_dirs()
    _save_list_file(LIBRARY_FILE, library)
    return str(LIBRARY_FILE)


def save_run_record(run_record: Dict[str, Any]) -> str:
    """Persist one run record and update run catalog."""
    ensure_data_dirs()
    run_id = run_record['run_id']
    file_path = OUTPUTS_DIR / f'run_{run_id}.json'
    file_path.write_text(json.dumps(run_record, indent=2), encoding='utf-8')

    catalog = load_run_catalog()
    catalog.append(
        {
            'run_id': run_id,
            'images_processed': run_record.get('images_processed', 0),
            'annotations_processed': run_record.get('annotations_processed', 0),
            'categories_count': run_record.get('categories_count', 0),
            'elapsed_ms': run_record.get('elapsed_ms', 0.0),
            'finished_at': run_record.get('finished_at', ''),
            'run_file': str(file_path),
        }
    )
    save_run_catalog(catalog)
    return str(file_path)


def save_annotation_record(annotation_record: Dict[str, Any], run_id: str) -> str:
    """Persist one annotation record and update annotation catalog."""
    ensure_data_dirs()
    annotation_id = annotation_record['annotation_id']
    file_path = OUTPUTS_DIR / f'ann_{annotation_id}_{run_id}.json'
    file_path.write_text(json.dumps(annotation_record, indent=2), encoding='utf-8')

    catalog = load_annotation_catalog()
    catalog.append(
        {
            'annotation_id': annotation_id,
            'run_id': run_id,
            'image_id': annotation_record.get('image_id', ''),
            'category': annotation_record.get('category', ''),
            'area': annotation_record.get('area', 0.0),
            'created_at': annotation_record.get('created_at', ''),
            'metadata_file': str(file_path),
        }
    )
    save_annotation_catalog(catalog)
    return str(file_path)


def load_json(filename: str):
    """Load JSON data from the local data directory."""
    ensure_data_dirs()
    path = DATA_DIR / filename
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []


def save_json(filename: str, data) -> None:
    """Save JSON data to the local data directory."""
    ensure_data_dirs()
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
