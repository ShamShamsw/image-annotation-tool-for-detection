"""
models.py - Data models for Project 36: Image Annotation Tool For Detection
"""

from datetime import datetime
from typing import Any, Dict, List


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.utcnow().isoformat(timespec='seconds') + 'Z'


def create_annotator_config(
    top_fields: int = 6,
    include_coco: bool = True,
    include_yolo: bool = True,
    include_voc: bool = True,
    include_area_metrics: bool = True,
    max_annotations_in_report: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Create a validated image annotation configuration record."""
    return {
        'project_type': 'image_annotation_tool_for_detection',
        'top_fields': max(1, int(top_fields)),
        'include_coco': bool(include_coco),
        'include_yolo': bool(include_yolo),
        'include_voc': bool(include_voc),
        'include_area_metrics': bool(include_area_metrics),
        'max_annotations_in_report': max(3, int(max_annotations_in_report)),
        'random_seed': int(random_seed),
        'created_at': _utc_timestamp(),
    }


def create_annotation_record(
    annotation_id: str,
    image_id: str,
    file_name: str,
    width: int,
    height: int,
    category: str,
    bbox_xywh: List[float],
    area: float,
    yolo_xywh: List[float],
    difficult: bool,
    truncated: bool,
) -> Dict[str, Any]:
    """Create one computed annotation record."""
    return {
        'annotation_id': annotation_id,
        'image_id': image_id,
        'file_name': file_name,
        'width': int(width),
        'height': int(height),
        'category': category,
        'bbox_xywh': [round(float(v), 4) for v in bbox_xywh],
        'area': round(float(area), 4),
        'yolo_xywh': [round(float(v), 6) for v in yolo_xywh],
        'difficult': bool(difficult),
        'truncated': bool(truncated),
        'created_at': _utc_timestamp(),
    }


def create_run_summary(
    run_id: str,
    images_processed: int,
    annotations_processed: int,
    categories_count: int,
    elapsed_ms: float,
    artifacts: Dict[str, Any],
    annotation_previews: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Create final run summary for reporting and persistence."""
    return {
        'run_id': run_id,
        'images_processed': int(images_processed),
        'annotations_processed': int(annotations_processed),
        'categories_count': int(categories_count),
        'elapsed_ms': float(elapsed_ms),
        'artifacts': artifacts,
        'annotation_previews': annotation_previews,
        'metrics': metrics,
        'finished_at': _utc_timestamp(),
    }


def create_record(**kwargs):
    """Backwards-compatible generic record factory."""
    return dict(kwargs)
