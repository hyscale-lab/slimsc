import csv
import sys

def check_suspicious_rows(csv_path):
    suspicious_rows = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row_idx, row in enumerate(reader, start=1):
            row_str = ",".join(row)
            if ',,' in row_str or '[]' in row_str or '"[]"' in row_str:
                suspicious_rows.append((row_idx, row_str))

    print(f"Total suspicious rows: {len(suspicious_rows)}")
    
    # 返回 True (exit code 1) 如果存在异常行，否则返回 0
    return 1 if suspicious_rows else 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_empty.py <csv_path>")
        sys.exit(2)

    csv_path = sys.argv[1]
    exit_code = check_suspicious_rows(csv_path)
    sys.exit(exit_code)
