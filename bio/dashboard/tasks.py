from celery import shared_task

@shared_task
def run_pixel_extraction(somevar):
    print("run_pixel_extraction")
    return True

@shared_task
def mul(x, y):
    return x * y

@shared_task
def xsum(numbers):
    return sum(numbers)
