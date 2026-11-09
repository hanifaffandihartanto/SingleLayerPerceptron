from pathlib import Path
import argparse
import csv
import openpyxl


def get_target(value):
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)

    label = str(value).strip().lower()
    if "setosa" in label:
        return 0
    if "versicolor" in label:
        return 1

    raise ValueError(
        f"Target '{value}' tidak dikenali. Gunakan Iris-setosa/"
        "Iris-versicolor atau angka 0/1."
    )


def read_dataset(file_path):
  
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {file_path}")

    workbook = openpyxl.load_workbook(
        file_path, data_only=True, read_only=True
    )

    try:
        worksheet = (
            workbook["Data"]
            if "Data" in workbook.sheetnames
            else workbook[workbook.sheetnames[0]]
        )

        dataset = []

        for row_number, row in enumerate(
            worksheet.iter_rows(min_col=1, max_col=5, values_only=True),
            start=1
        ):
            if all(value is None for value in row):
                continue

            # Lewati header kolom jika ada.
            first_value = str(row[0]).strip().lower()
            if row_number == 1 and first_value in {
                "sepal length",
                "sepal length (cm)",
                "sepal_length",
                "x1"
            }:
                continue

            if any(value is None for value in row):
                raise ValueError(
                    f"Data tidak lengkap pada {file_path.name}, "
                    f"baris {row_number}."
                )

            try:
                features = [float(value) for value in row[:4]]
            except (TypeError, ValueError):
                if not dataset:
                    # Mengizinkan satu baris header dengan nama kolom lain.
                    continue
                raise ValueError(
                    f"Fitur harus numerik pada {file_path.name}, "
                    f"baris {row_number}."
                )

            target = get_target(row[4])

            dataset.append({
                "source_row": row_number,
                "x": features,
                "y": target
            })

        if not dataset:
            raise ValueError(
                f"Tidak ada data yang terbaca dari {file_path.name}."
            )

        return dataset
    finally:
        workbook.close()