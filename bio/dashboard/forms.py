from django import forms
from .models import Report


class RoiSetForm(forms.Form):
    title = forms.CharField(label='Label', max_length=50)
    file = forms.FileField(label='RoiSet File')


class ExampleModelForm(forms.Form):
    report1 = forms.ModelChoiceField(
        label="Report 1",
        queryset = Report.objects.all(),
        required=True
    )

    report2 = forms.ModelChoiceField(
        label="Report 2",
        queryset = Report.objects.all(),
        required=False
    )

    report3 = forms.ModelChoiceField(
        label="Report 3",
        queryset = Report.objects.all(),
        required=False
    )

    report4 = forms.ModelChoiceField(
        label="Report 4",
        queryset = Report.objects.all(),
        required=False
    )

    report5 = forms.ModelChoiceField(
        label="Report 5",
        queryset = Report.objects.all(),
        required=False
    )

    report6 = forms.ModelChoiceField(
        label="Report 6",
        queryset = Report.objects.all(),
        required=False
    )
