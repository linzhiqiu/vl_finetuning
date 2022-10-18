from engine.templates.default import DEFAULT_TEMPLATES
from engine.templates.extra import EXTRA_TEMPLATES
from engine.templates.ensemble import ENSEMBLE_TEMPLATES
from engine.templates.ensemble_all import ENSEMBLE_ALL_TEMPLATES
from engine.templates.ensemble_same import ENSEMBLE_SAME_TEMPLATES
from engine.templates.tip_adapter import TIP_ADAPTER_TEMPLATES

def get_templates(dataset_name, template_name):
    """Return a list of templates to use for the given config."""
    if template_name == 'classname':
        return ["{}"]
    elif template_name == 'single':
        return ["a photo of a {}."]
    elif template_name == 'default':
        return DEFAULT_TEMPLATES[dataset_name]
    elif template_name == 'tip_adapter':
        return TIP_ADAPTER_TEMPLATES[dataset_name]
    elif template_name == 'extra':
        if EXTRA_TEMPLATES[dataset_name] is None:
            raise ValueError('No extra templates for {}'.format(dataset_name))
        return EXTRA_TEMPLATES[dataset_name]
    elif template_name == 'extra_default':
        if EXTRA_TEMPLATES[dataset_name] is None:
            raise ValueError('No extra templates for {}'.format(dataset_name))
        return EXTRA_TEMPLATES[dataset_name] + ["a photo of a {}."]
    elif template_name == 'ensemble':
        return ENSEMBLE_TEMPLATES
    elif template_name == 'ensemble_all':
        return ENSEMBLE_ALL_TEMPLATES[dataset_name]
    elif template_name == 'ensemble_same':
        return ENSEMBLE_SAME_TEMPLATES
    else:
        raise ValueError('Unknown template: {}'.format(template_name))