# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# --------
# Licensed under the MIT license.
# --------
# Author: Weihao Cui
import argparse


# generate start_ids.csv, $request_bs lines, each line has $input_len number, each number is a random number between 0 and 50256
def generate_start_ids(request_bs, input_len, id_range=50255, used_gpus=8):
    import random
    import csv

    file_name = f"start_ids_{used_gpus}.csv"
    with open(file_name, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for i in range(request_bs):
            csv_writer.writerow([random.randint(0, id_range) for _ in range(input_len)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request_bs", type=int, default=8, help="request batch size")
    parser.add_argument("--input_len", type=int, default=8, help="input length")
    parser.add_argument("--used_gpus", type=int, default=8, help="used gpus")
    args = parser.parse_args()
    generate_start_ids(args.request_bs, args.input_len, used_gpus=args.used_gpus)
