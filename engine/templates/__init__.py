from engine.templates.default import DEFAULT_TEMPLATES
from engine.templates.extra import EXTRA_TEMPLATES

def get_templates(dataset_name, template_name):
    """Return a list of templates to use for the given config."""
    if template_name == 'classname':
        return ["{}"]
    elif template_name == 'single':
        return ["a photo of a {}."]
    elif template_name == 'default':
        return DEFAULT_TEMPLATES[dataset_name]
    elif template_name == 'extra':
        if EXTRA_TEMPLATES[dataset_name] is None:
            raise ValueError('No extra templates for {}'.format(dataset_name))
        return EXTRA_TEMPLATES[dataset_name]
    else:
        raise ValueError('Unknown template: {}'.format(template_name))