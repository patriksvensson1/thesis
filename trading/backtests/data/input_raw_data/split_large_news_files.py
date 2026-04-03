import csv
from pathlib import Path

# This script splits large raw GDELT files into two parts so they can be stored in the repository despite file-size limits.

input_file = "trading\\backtests\data\\input_raw_data\\raw_gdelt_news_2024.csv"  # change this to your file path

input_path = Path(input_file)
output_1 = input_path.with_name(f"{input_path.stem}_1.csv")
output_2 = input_path.with_name(f"{input_path.stem}_2.csv")

with input_path.open("r", newline="", encoding="utf-8") as f:
    reader = list(csv.reader(f))

header = reader[0]
rows = reader[1:]

mid = len(rows) // 2

first_half = rows[:mid]
second_half = rows[mid:]

with output_1.open("w", newline="", encoding="utf-8") as f1:
    writer = csv.writer(f1)
    writer.writerow(header)
    writer.writerows(first_half)

with output_2.open("w", newline="", encoding="utf-8") as f2:
    writer = csv.writer(f2)
    writer.writerow(header)
    writer.writerows(second_half)

print(f"Created: {output_1}")
print(f"Created: {output_2}")