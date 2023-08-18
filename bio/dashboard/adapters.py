from django.conf import settings
from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.http import HttpResponse

from allauth.exceptions import ImmediateHttpResponse
from django import forms

class CustomAccountAdapter(DefaultAccountAdapter):

    def clean_email(self, email):
        """
        Validates an email value. You can hook into this if you want to
        (dynamically) restrict what email addresses can be chosen.
        """

        if not email.endswith("@454.bio"):
            raise forms.ValidationError(f"{email} not valid")

        return super().clean_email(email)

    def is_open_for_signup(self, request):
        return True  # No email/password signups allowed

class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):

        email_domain = sociallogin.user.email.split('@')[1].lower()
        if not email_domain == "454.bio":
            raise ImmediateHttpResponse(HttpResponse(sociallogin.user.email + ' is not valid member of 454'))
        else:
            pass
