from celery import shared_task
import ziontools
import os
import subprocess
import shlex

@shared_task
def run_pixel_extraction(raw_path, report_path, roiset_zip_filename):
    print("run_pixel_extraction")

    full_roiset_path = os.path.join(report_path, roiset_zip_filename)
    full_spot_pixel_data_path = os.path.join(report_path, "spot_pixel_data.csv")

    ziontools.extract_roiset_pixel_data(
        raw_path,
        full_roiset_path,
        full_spot_pixel_data_path,
        200,  # max_number_of_pixel_per_spot=200,
        0     # start_645_image_number=0 auto
    )

    df = ziontools.calculate_and_apply_transformation(
        full_spot_pixel_data_path,
        full_roiset_path,
        raw_path,
        report_path,
        {},
        {}
    )

    return 9


@shared_task
def mul(x, y):
    return x * y


@shared_task
def xsum(numbers):
    return sum(numbers)
