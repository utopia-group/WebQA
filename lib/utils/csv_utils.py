import csv
import sys


csv.field_size_limit(sys.maxsize)


def save_dict_to_csv(path, records):
    if len(records) == 0:
        return

    keys = records[0].keys()
    with open(path, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(records)


def read_csv_to_dict(path):
    records = []
    with open(path) as input_file:
        rd = csv.DictReader(input_file, delimiter=',', skipinitialspace=True)
        for row in rd:
            records.append(row)
    return records
