"""
display.py - Presentation helpers for Project 36: Image Annotation Tool For Detection
"""

from typing import Any, Dict, List


def format_header() -> str:
    """Format session header banner."""
    return '=' * 70 + '\n' + '   IMAGE ANNOTATION TOOL FOR DETECTION\n' + '=' * 70


def format_startup_guide(config: Dict[str, Any], profile: Dict[str, Any]) -> str:
    """Format startup configuration and historical profile."""
    recent = ', '.join(profile.get('recent_annotations', [])) or 'None yet'
    lines = [
        '',
        'Configuration:',
        f"   Project type:         {config['project_type']}",
        '   Annotation engine:    bbox -> COCO/YOLO/VOC (stdlib only)',
        f"   Top fields:           {config['top_fields']}",
        f"   Include COCO:         {config['include_coco']}",
        f"   Include YOLO:         {config['include_yolo']}",
        f"   Include VOC:          {config['include_voc']}",
        f"   Include area metrics: {config['include_area_metrics']}",
        f"   Max ann. report:      {config['max_annotations_in_report']}",
        f"   Random seed:          {config['random_seed']}",
        '',
        'Startup:',
        '   Data directory:       data/',
        '   Outputs directory:    data/outputs/',
        (
            f"   Annotation library:   {profile['library_file']} "
            f"(loaded {profile['library_available']} images)"
        ),
        (
            f"   Run catalog:          {profile['catalog_file']} "
            f"(loaded {profile['runs_stored']} runs)"
        ),
        (
            f"   Annotation catalog:   {profile['annotation_catalog_file']} "
            f"(loaded {profile['annotation_records_stored']} records)"
        ),
        f'   Recent annotations:   {recent}',
        '',
        '---',
    ]
    return '\n'.join(lines)


def format_annotation_table(previews: List[Dict[str, Any]]) -> str:
    """Format annotation property table."""
    if not previews:
        return 'No annotation data available.'
    lines = [
        'Annotation previews:',
        '   ID      | Image   | Category      | BBox (x,y,w,h)          | Area(px2) | YOLO(cx,cy,w,h)',
        '   --------+---------+---------------+--------------------------+-----------+-----------------',
    ]
    for ann in previews:
        x, y, w, h = ann.get('bbox_xywh', [0, 0, 0, 0])
        cx, cy, nw, nh = ann.get('yolo_xywh', [0.0, 0.0, 0.0, 0.0])
        lines.append(
            '   '
            f"{ann['annotation_id'][:7]:<7} | "
            f"{ann['image_id'][:7]:<7} | "
            f"{ann.get('category', '')[:13]:<13} | "
            f"[{x:>4.0f},{y:>4.0f},{w:>4.0f},{h:>4.0f}]           | "
            f"{ann.get('area', 0.0):>9.0f} | "
            f"{cx:>.3f},{cy:>.3f},{nw:>.3f},{nh:>.3f}"
        )
    return '\n'.join(lines)


def format_run_report(summary: Dict[str, Any]) -> str:
    """Format final run report."""
    artifacts = summary.get('artifacts', {})
    metrics = summary.get('metrics', {})
    metadata_files = artifacts.get('metadata_files', [])
    lines = [
        '',
        'Run complete:',
        f"   Run ID:               {summary['run_id']}",
        f"   Images processed:     {summary['images_processed']}",
        f"   Annotations processed: {summary['annotations_processed']}",
        f"   Categories found:     {summary['categories_count']}",
        f"   Elapsed time:         {summary['elapsed_ms']:.2f} ms",
        '',
        (
            f"Dataset metrics: mean_area={metrics.get('mean_box_area', 0.0):.2f} | "
            f"min_area={metrics.get('min_box_area', 0.0):.2f} | "
            f"max_area={metrics.get('max_box_area', 0.0):.2f} | "
            f"mean_ann/img={metrics.get('mean_annotations_per_image', 0.0):.2f} | "
            f"largest_cls_share={metrics.get('largest_category_share', 0.0):.0%}"
        ),
        '',
        format_annotation_table(summary.get('annotation_previews', [])),
        '',
        'Artifacts saved:',
        f"   Run record:           {artifacts.get('run_file', 'N/A')}",
        f"   COCO export:          {artifacts.get('coco_file', 'N/A')}",
        f"   YOLO export:          {artifacts.get('yolo_file', 'N/A')}",
        f"   VOC export dir:       {artifacts.get('voc_dir', 'N/A')}",
        f"   Metadata exports:     {len(metadata_files)}",
    ]
    return '\n'.join(lines)


def format_message(message: str) -> str:
    """Format a user-facing message string."""
    return f'[Project 36] {message}'
