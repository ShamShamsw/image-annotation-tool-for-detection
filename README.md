# Beginner Project 36: Image Annotation Tool For Detection

**Time:** 4-6 hours  
**Difficulty:** Intermediate Beginner  
**Focus:** Bounding-box validation, multi-format export (COCO/YOLO/VOC), and persistent detection annotation analytics

---

## Why This Project?

Object detection models are only as good as the annotation data used to train them. This project demonstrates how to build a practical annotation pipeline in pure Python, where image metadata and bounding boxes are normalized and exported in standard formats.

This project teaches end-to-end detection dataset workflow concepts where you can:

- load or auto-generate a reusable local image annotation library,
- validate and clip bounding boxes against image boundaries,
- compute area metrics from pixel-space bounding boxes,
- convert annotations from XYWH pixel coordinates to YOLO normalized format,
- export structured COCO-style JSON with image, annotation, and category blocks,
- generate Pascal VOC XML files per image with object bounding boxes,
- persist run summaries and per-annotation metadata to JSON,
- compute dataset-level quality metrics from annotation distributions,
- track recent annotations and historical run catalogs,
- and print a readable terminal report with preview rows and export paths.

---

## More Projects

You can access this project and more in this separate repository:

[student-interview-prep](https://github.com/ShamShamsw/student-interview-prep.git)

---

## What You Will Build

You will build an image annotation tool for detection that:

1. Loads an annotation library from `data/library.json` (or seeds a starter set).
2. Parses image records with file name, dimensions, and object lists.
3. Validates and clips each bounding box to legal image coordinates.
4. Converts pixel-space `x,y,w,h` to normalized YOLO `cx,cy,w,h`.
5. Computes per-annotation area and basic quality flags.
6. Builds a category index for compact numeric class IDs.
7. Exports COCO JSON (`images`, `annotations`, `categories`) for downstream tooling.
8. Exports YOLO label rows for lightweight training workflows.
9. Exports Pascal VOC XML files for interoperability with VOC-compatible tools.
10. Persists per-annotation metadata and run summaries for history and auditing.

---

## Requirements

- Python 3.11+
- No external dependencies - the annotation engine uses stdlib only.

---

## Example Session

```text
======================================================================
   IMAGE ANNOTATION TOOL FOR DETECTION
======================================================================

Configuration:
   Project type:         image_annotation_tool_for_detection
   Annotation engine:    bbox -> COCO/YOLO/VOC (stdlib only)
   Top fields:           6
   Include COCO:         True
   Include YOLO:         True
   Include VOC:          True
   Include area metrics: True
   Max ann. report:      10
   Random seed:          42

Startup:
   Data directory:       data/
   Outputs directory:    data/outputs/
   Annotation library:   data/library.json (loaded 6 images)
   Run catalog:          data/runs.json (loaded 0 runs)
   Annotation catalog:   data/annotations.json (loaded 0 records)
   Recent annotations:   None yet

---

Run complete:
   Run ID:               20260317_120000
   Images processed:     6
   Annotations processed:16
   Categories found:     7
   Elapsed time:         4.87 ms

Dataset metrics: mean_area=34756.50 | min_area=7696.00 | max_area=175840.00 | mean_ann/img=2.67 | largest_cls_share=31%

Annotation previews:
   ID      | Image   | Category      | BBox (x,y,w,h)          | Area(px2) | YOLO(cx,cy,w,h)
   --------+---------+---------------+--------------------------+-----------+-----------------
   ann_001 | img_001 | car           | [ 110, 320, 260, 180]   |     46800 | 0.188,0.569,0.203,0.250
   ann_002 | img_001 | person        | [ 460, 260,  80, 220]   |     17600 | 0.391,0.514,0.062,0.306
   ann_003 | img_002 | person        | [ 210, 410,  96, 280]   |     26880 | 0.134,0.509,0.050,0.259

Artifacts saved:
   Run record:           data/outputs/run_20260317_120000.json
   COCO export:          data/outputs/coco_20260317_120000.json
   YOLO export:          data/outputs/yolo_20260317_120000.txt
   VOC export dir:       data/outputs/voc_20260317_120000/
   Metadata exports:     16
```

---

## Run

```bash
python main.py
```
