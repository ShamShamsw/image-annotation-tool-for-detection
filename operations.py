"""
operations.py - Business logic for Project 36: Image Annotation Tool For Detection
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Tuple

from models import create_annotation_record, create_annotator_config, create_run_summary
from storage import (
    OUTPUTS_DIR,
    ensure_data_dirs,
    load_annotation_catalog,
    load_library,
    load_run_catalog,
    save_annotation_record,
    save_library,
    save_run_record,
)


def _run_id() -> str:
    """Build a compact run ID from UTC timestamp."""
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def _clip_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Clip bounding box to image bounds while preserving non-negative area."""
    x1 = max(0.0, min(float(x), float(width)))
    y1 = max(0.0, min(float(y), float(height)))
    x2 = max(x1, min(x1 + max(0.0, float(w)), float(width)))
    y2 = max(y1, min(y1 + max(0.0, float(h)), float(height)))
    return x1, y1, x2 - x1, y2 - y1


def _to_yolo_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> List[float]:
    """Convert pixel xywh box into normalized YOLO center-width-height format."""
    cx = (x + w / 2.0) / float(width)
    cy = (y + h / 2.0) / float(height)
    nw = w / float(width)
    nh = h / float(height)
    return [cx, cy, nw, nh]


def _default_library() -> List[Dict[str, Any]]:
    """Return deterministic starter image annotation library used on first run."""
    return [
        {
            'image_id': 'img_001',
            'file_name': 'street_001.jpg',
            'width': 1280,
            'height': 720,
            'objects': [
                {'category': 'car', 'bbox_xywh': [110, 320, 260, 180], 'difficult': False, 'truncated': False},
                {'category': 'person', 'bbox_xywh': [460, 260, 80, 220], 'difficult': False, 'truncated': False},
            ],
        },
        {
            'image_id': 'img_002',
            'file_name': 'crosswalk_014.jpg',
            'width': 1920,
            'height': 1080,
            'objects': [
                {'category': 'person', 'bbox_xywh': [210, 410, 96, 280], 'difficult': False, 'truncated': False},
                {'category': 'person', 'bbox_xywh': [640, 430, 94, 268], 'difficult': False, 'truncated': False},
                {'category': 'traffic_light', 'bbox_xywh': [1540, 80, 52, 148], 'difficult': True, 'truncated': False},
            ],
        },
        {
            'image_id': 'img_003',
            'file_name': 'parkinglot_007.jpg',
            'width': 1600,
            'height': 900,
            'objects': [
                {'category': 'car', 'bbox_xywh': [98, 372, 220, 146], 'difficult': False, 'truncated': False},
                {'category': 'car', 'bbox_xywh': [402, 360, 232, 155], 'difficult': False, 'truncated': False},
                {'category': 'truck', 'bbox_xywh': [820, 320, 360, 220], 'difficult': False, 'truncated': True},
            ],
        },
        {
            'image_id': 'img_004',
            'file_name': 'bike_lane_022.jpg',
            'width': 1366,
            'height': 768,
            'objects': [
                {'category': 'bicycle', 'bbox_xywh': [540, 318, 180, 196], 'difficult': False, 'truncated': False},
                {'category': 'person', 'bbox_xywh': [565, 248, 98, 248], 'difficult': False, 'truncated': False},
            ],
        },
        {
            'image_id': 'img_005',
            'file_name': 'bus_stop_003.jpg',
            'width': 1280,
            'height': 720,
            'objects': [
                {'category': 'bus', 'bbox_xywh': [300, 210, 560, 314], 'difficult': False, 'truncated': False},
                {'category': 'person', 'bbox_xywh': [180, 286, 86, 218], 'difficult': False, 'truncated': False},
                {'category': 'person', 'bbox_xywh': [960, 298, 76, 208], 'difficult': False, 'truncated': False},
            ],
        },
        {
            'image_id': 'img_006',
            'file_name': 'intersection_019.jpg',
            'width': 2048,
            'height': 1152,
            'objects': [
                {'category': 'car', 'bbox_xywh': [940, 510, 270, 170], 'difficult': False, 'truncated': False},
                {'category': 'motorcycle', 'bbox_xywh': [700, 560, 160, 130], 'difficult': False, 'truncated': False},
                {'category': 'traffic_light', 'bbox_xywh': [300, 90, 58, 152], 'difficult': False, 'truncated': False},
            ],
        },
    ]


def _build_voc_xml(image: Dict[str, Any], objects: List[Dict[str, Any]]) -> str:
    """Create a minimal Pascal VOC XML document for one image."""
    lines = [
        '<annotation>',
        f"  <filename>{image['file_name']}</filename>",
        '  <size>',
        f"    <width>{image['width']}</width>",
        f"    <height>{image['height']}</height>",
        '    <depth>3</depth>',
        '  </size>',
    ]
    for obj in objects:
        x, y, w, h = obj['bbox_xywh']
        xmin = int(round(x))
        ymin = int(round(y))
        xmax = int(round(x + w))
        ymax = int(round(y + h))
        lines.extend(
            [
                '  <object>',
                f"    <name>{obj['category']}</name>",
                f"    <difficult>{1 if obj['difficult'] else 0}</difficult>",
                f"    <truncated>{1 if obj['truncated'] else 0}</truncated>",
                '    <bndbox>',
                f'      <xmin>{xmin}</xmin>',
                f'      <ymin>{ymin}</ymin>',
                f'      <xmax>{xmax}</xmax>',
                f'      <ymax>{ymax}</ymax>',
                '    </bndbox>',
                '  </object>',
            ]
        )
    lines.append('</annotation>')
    return '\n'.join(lines)


def load_annotator_profile() -> Dict[str, Any]:
    """Return startup profile built from previously saved catalogs."""
    runs = load_run_catalog()
    annotations = load_annotation_catalog()
    library = load_library()
    recent_annotations = [
        f"{item.get('annotation_id', '')}:{item.get('category', '')}:{item.get('area', 0.0):.0f}px2"
        for item in annotations[-8:]
    ]
    return {
        'catalog_file': 'data/runs.json',
        'annotation_catalog_file': 'data/annotations.json',
        'library_file': 'data/library.json',
        'runs_stored': len(runs),
        'annotation_records_stored': len(annotations),
        'library_available': len(library),
        'recent_annotations': recent_annotations,
    }


def run_core_flow() -> Dict[str, Any]:
    """Run one complete image annotation processing session."""
    ensure_data_dirs()
    config = create_annotator_config()

    run_id = _run_id()
    started = time.perf_counter()

    library = load_library()
    if not library:
        library = _default_library()
        save_library(library)

    category_to_id: Dict[str, int] = {}
    image_to_voc_objects: Dict[str, List[Dict[str, Any]]] = {}
    computed: List[Dict[str, Any]] = []
    metadata_files: List[str] = []
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    yolo_rows: List[str] = []

    annotation_index = 1
    for image in library:
        image_id = image['image_id']
        file_name = image['file_name']
        width = int(image['width'])
        height = int(image['height'])
        objects = image.get('objects', [])

        coco_images.append(
            {
                'id': image_id,
                'file_name': file_name,
                'width': width,
                'height': height,
            }
        )

        image_to_voc_objects[image_id] = []

        for obj in objects:
            category = str(obj.get('category', 'unknown'))
            raw_box = obj.get('bbox_xywh', [0, 0, 0, 0])
            x, y, w, h = _clip_bbox(
                raw_box[0], raw_box[1], raw_box[2], raw_box[3], width, height
            )
            area = w * h
            yolo_xywh = _to_yolo_bbox(x, y, w, h, width, height)
            difficult = bool(obj.get('difficult', False))
            truncated = bool(obj.get('truncated', False))

            annotation_id = f'ann_{annotation_index:03d}'
            annotation_index += 1

            record = create_annotation_record(
                annotation_id=annotation_id,
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                category=category,
                bbox_xywh=[x, y, w, h],
                area=area,
                yolo_xywh=yolo_xywh,
                difficult=difficult,
                truncated=truncated,
            )
            computed.append(record)
            metadata_files.append(save_annotation_record(record, run_id))

            if category not in category_to_id:
                category_to_id[category] = len(category_to_id) + 1
            category_id = category_to_id[category]

            coco_annotations.append(
                {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
                    'area': round(area, 4),
                    'iscrowd': 0,
                }
            )

            yolo_rows.append(
                f"{image_id} {category_id - 1} "
                f"{yolo_xywh[0]:.6f} {yolo_xywh[1]:.6f} "
                f"{yolo_xywh[2]:.6f} {yolo_xywh[3]:.6f}"
            )

            image_to_voc_objects[image_id].append(
                {
                    'category': category,
                    'bbox_xywh': [x, y, w, h],
                    'difficult': difficult,
                    'truncated': truncated,
                }
            )

    computed.sort(key=lambda r: (r['image_id'], -r['area']))

    coco_categories = [
        {'id': category_id, 'name': category}
        for category, category_id in sorted(category_to_id.items(), key=lambda kv: kv[1])
    ]

    coco_file = OUTPUTS_DIR / f'coco_{run_id}.json'
    coco_file.write_text(
        json.dumps(
            {
                'run_id': run_id,
                'images': coco_images,
                'annotations': coco_annotations,
                'categories': coco_categories,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    yolo_file = OUTPUTS_DIR / f'yolo_{run_id}.txt'
    yolo_file.write_text('\n'.join(yolo_rows), encoding='utf-8')

    voc_dir = OUTPUTS_DIR / f'voc_{run_id}'
    voc_dir.mkdir(parents=True, exist_ok=True)
    for image in library:
        image_id = image['image_id']
        xml = _build_voc_xml(image, image_to_voc_objects.get(image_id, []))
        xml_path = voc_dir / f"{Path(image['file_name']).stem}.xml"
        xml_path.write_text(xml, encoding='utf-8')

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    areas = [item['area'] for item in computed]
    category_counts = Counter(item['category'] for item in computed)

    metrics: Dict[str, Any] = {
        'mean_box_area': round(statistics.mean(areas), 4) if areas else 0.0,
        'min_box_area': round(min(areas), 4) if areas else 0.0,
        'max_box_area': round(max(areas), 4) if areas else 0.0,
        'mean_annotations_per_image': (
            round(len(computed) / len(library), 4) if library else 0.0
        ),
        'largest_category_share': (
            round(max(category_counts.values()) / len(computed), 4)
            if computed
            else 0.0
        ),
    }

    artifacts: Dict[str, Any] = {
        'run_file': str(OUTPUTS_DIR / f'run_{run_id}.json'),
        'coco_file': str(coco_file),
        'yolo_file': str(yolo_file),
        'voc_dir': str(voc_dir),
        'metadata_files': metadata_files,
    }

    previews = computed[: config['max_annotations_in_report']]
    summary = create_run_summary(
        run_id=run_id,
        images_processed=len(library),
        annotations_processed=len(computed),
        categories_count=len(category_to_id),
        elapsed_ms=elapsed_ms,
        artifacts=artifacts,
        annotation_previews=previews,
        metrics=metrics,
    )

    save_run_record(summary)
    return summary
