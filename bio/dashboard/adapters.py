from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.http import HttpResponse

from allauth.exceptions import ImmediateHttpResponse

class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):

        email_domain = sociallogin.user.email.split('@')[1].lower()
        if not email_domain == "454.bio":
            raise ImmediateHttpResponse(HttpResponse(sociallogin.user.email + ' is not valid member of 454'))
        else:
            pass
