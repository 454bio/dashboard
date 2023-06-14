from django.contrib import admin

from .models import Device, Run, Report, Reservoir
admin.site.register(Device)
admin.site.register(Run)
admin.site.register(Report)
admin.site.register(Reservoir)
