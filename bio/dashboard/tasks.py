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

    # ziontools.extract_roiset_pixel_data(
    #     raw_path,
    #     full_roiset_path,
    #     full_spot_pixel_data_path,
    #     200,  # max_number_of_pixel_per_spot=200,
    #     0     # start_645_image_number=0 auto
    # )
    #
    # df = ziontools.calculate_and_apply_transformation(
    #     full_spot_pixel_data_path,
    #     full_roiset_path,
    #     raw_path,
    #     report_path,
    #     {},
    #     {}
    # )


    with open(os.path.join(report_path, 'pipeline_out.txt'), 'w') as fd:

        cmd = shlex.split(
            f"/opt/dashboard/venv/bin/python /opt/tools_playground/wrapper/extract_roiset_pixel_data.py -i {raw_path} -r {full_roiset_path} -o {full_spot_pixel_data_path}"
        )
        print(cmd)
        process = subprocess.Popen(
            cmd,
            cwd=report_path,
            stdout=fd,
            stderr=subprocess.STDOUT
        )
        output = process.communicate()[0]
        ret = process.wait()

        cmd = shlex.split(
            f"/opt/dashboard/venv/bin/python /opt/tools_playground/wrapper/color_transformation.py -i {raw_path} -r {full_roiset_path} -p {full_spot_pixel_data_path} -o {report_path}"
        )
        print(cmd)
        process = subprocess.Popen(
            cmd,
            cwd=report_path,
            stdout=fd,
            stderr=subprocess.STDOUT
        )
        output = process.communicate()[0]
        ret = process.wait()

        cmd = shlex.split(
            f"/opt/dashboard/venv/bin/python /opt/tools_playground/wrapper/dephaser.py -i {os.path.join(report_path,'color_transformed_spots.csv')} -o {os.path.join(report_path, 'basecalls.csv')}"
        )
        print(cmd)
        process = subprocess.Popen(
            cmd,
            cwd=report_path,
            stdout=fd,
            stderr=subprocess.STDOUT
        )
        output = process.communicate()[0]
        ret = process.wait()

        cmd = shlex.split("ls -l")
        print(cmd)
        process = subprocess.Popen(
            cmd,
            cwd=report_path,
            stdout=fd,
            stderr=subprocess.STDOUT
        )
        output = process.communicate()[0]
        ret = process.wait()

    return 9


@shared_task
def mul(x, y):
    return x * y


@shared_task
def xsum(numbers):
    return sum(numbers)
