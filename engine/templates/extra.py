imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

flowers_templates = [
    'a photo of a {}, a type of flower.',
]

aircraft_templates = [
    'a photo of a {}, a type of aircraft.',
    'a photo of the {}, a type of aircraft.',
]


food_templates = [
    'a photo of {}, a type of food.',
]

pets_templates = [
    'a photo of a {}, a type of pet.',
]

cars_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    "a photo of my {}.",
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

dtd_templates = [
    'a photo of a {} texture.',
    'a photo of a {} pattern.',
    'a photo of a {} thing.',
    'a photo of a {} object.',
    'a photo of the {} texture.',
    'a photo of the {} pattern.',
    'a photo of the {} thing.',
    'a photo of the {} object.',
]


caltech101_templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.'
]

ucf101_templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]

sun397_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
]

eurosat_templates = [
    'a centered satellite photo of {}.',
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
]

EXTRA_TEMPLATES = {
    "oxford_pets": None,
    "oxford_flowers": None,
    "fgvc_aircraft": aircraft_templates,
    "dtd": dtd_templates,
    "eurosat": eurosat_templates,
    "stanford_cars": cars_templates,
    "food101": None,
    "sun397": sun397_templates,
    "caltech101": caltech101_templates,
    "ucf101": ucf101_templates,
    "imagenet": imagenet_templates,
    "imagenet_sketch": imagenet_templates,
    "imagenetv2": imagenet_templates,
    "imagenet_a": imagenet_templates,
    "imagenet_r": imagenet_templates,
}