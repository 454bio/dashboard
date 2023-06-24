from django.db import models
from django.urls import reverse

class Device(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=200)
    serial_number = models.CharField(max_length=20)
    storage = models.CharField(max_length=30, null=True, blank=True)
    notes = models.CharField(max_length=30, null=True, blank=True)
    ip = models.CharField(max_length=40, null=True, blank=True)
    filter = models.CharField(max_length=40, null=True, blank=True)
    tempsensor = models.CharField(max_length=40, null=True, blank=True)

    def get_absolute_url(self):
        return reverse('device-detail', args=[str(self.id)])

    def __str__(self):
        return self.name + " " + self.serial_number


class Reservoir(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    assembly_date = models.DateField(null=True, blank=True)
    serial_number = models.IntegerField(null=False, unique=True)
    vendor = models.CharField(max_length=200)
    substrate = models.CharField(max_length=200)
    cleaning = models.CharField(max_length=200)
    coating = models.CharField(max_length=200)
    functionalization = models.CharField(max_length=200)
    library = models.CharField(max_length=200)
    blocking = models.CharField(max_length=200)

    def get_absolute_url(self):
        return reverse('reservoir-detail', args=[str(self.id)])

    def __str__(self):
        return str(f"{self.serial_number} {self.substrate} {self.functionalization} {self.library}")


class Run(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=200, unique=True)
    device = models.ForeignKey(Device, null=True, blank=True, on_delete=models.SET_NULL)
    operator = models.CharField(max_length=50, null=True, blank=True)
    sequencing_protocol = models.CharField(max_length=100, null=True, blank=True)
    path = models.CharField(max_length=200, null=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    notes = models.CharField(max_length=400, null=True, blank=True)
    reservoir = models.ForeignKey(Reservoir, null=True, blank=True, on_delete=models.SET_NULL)

    def get_absolute_url(self):
        return reverse('run-detail', args=[str(self.id)])

    def __str__(self):
        return self.name



class Report(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=200)
    run = models.ForeignKey(Run, null=True, blank=True, on_delete=models.SET_NULL)
    path = models.CharField(max_length=200, null=True)

    def get_absolute_url(self):
        return reverse('report-detail', args=[str(self.id)])

    def __str__(self):
        return self.name

