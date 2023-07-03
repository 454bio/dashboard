from django import forms

class RoiSetForm(forms.Form):
    title = forms.CharField(label='Label', max_length=50)
    file = forms.FileField()
