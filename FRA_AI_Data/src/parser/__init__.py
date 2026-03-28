"""FRA file parsing: CSV, Excel, and format auto-detection."""

from src.parser.auto_parser import detect_columns, load_fra_data
from src.parser.universal_parser import (
    FRAParseError,
    detect_fra_columns,
    parse_fra_file,
    parse_fra_file_safe,
)

__all__ = [
    "FRAParseError",
    "detect_columns",
    "detect_fra_columns",
    "load_fra_data",
    "parse_fra_file",
    "parse_fra_file_safe",
]
