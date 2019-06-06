from .xml_style import XMLDataset


class VOCDataset(XMLDataset):

    CLASSES = ['spike']

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
