"""
main.py - Entry point for Project 36: Image Annotation Tool For Detection
"""

from display import format_header, format_run_report, format_startup_guide
from models import create_annotator_config
from operations import load_annotator_profile, run_core_flow
from storage import ensure_data_dirs


def main() -> None:
    """Run one complete annotation session."""
    ensure_data_dirs()
    print(format_header())

    config = create_annotator_config()
    profile = load_annotator_profile()
    print(format_startup_guide(config, profile))

    summary = run_core_flow()
    print(format_run_report(summary))


if __name__ == '__main__':
    main()
