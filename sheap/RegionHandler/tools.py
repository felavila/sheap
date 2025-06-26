from typing import Any, Dict, List, Optional, Tuple, Union


def known_tied_relations():
    [
        (
            ('OIIIb', 'OIIIc'),
            ['amplitude_OIIIb_component_narrow', 'amplitude_OIIIc_component_narrow', '*0.3'],
        ),
        (
            ('NIIa', 'NIIb'),
            ['amplitude_NIIa_component_narrow', 'amplitude_NIIb_component_narrow', '*0.3'],
        ),
        (('NIIa', 'NIIb'), ['center_NIIa_component_narrow', 'center_NIIb_component_narrow']),
        (
            ('OIIIb', 'OIIIc'),
            ['center_OIIIb_component_narrow', 'center_OIIIc_component_narrow'],
        ),
    ]