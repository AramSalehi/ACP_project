#%%
from pathlib import Path
import re

data_dir = Path("raw_data/data")

zero_area_files = []

for info_file in sorted(data_dir.glob("*_INFO.txt")):
    text = info_file.read_text(encoding="utf-8", errors="replace")

    values = {}

    for field in ["caarea", "cparea"]:
        match = re.search(
            rf"(?:^|[\t{{]){field}\s*=\s*([^\t}}\r\n]+)",
            text,
        )

        if match:
            raw_value = match.group(1).strip()

            try:
                values[field] = float(raw_value)
            except ValueError:
                values[field] = None

    zero_fields = [
        field
        for field, value in values.items()
        if value == 0
    ]

    if zero_fields:
        zero_area_files.append(
            {
                "file": info_file.name,
                "zero_fields": zero_fields,
            }
        )

        print(
            f"{info_file.name}: "
            f"{', '.join(zero_fields)} = 0"
        )

print(f"\nTotal INFO files with zero area: {len(zero_area_files)}")