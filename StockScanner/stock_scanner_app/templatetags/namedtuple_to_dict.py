from django import template

register = template.Library()

@register.filter
def namedtuple_to_dict(namedtuple_instance):
    return namedtuple_instance._asdict()

@register.filter
def sort_dict(d):
    return sorted(d.items())
