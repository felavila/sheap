from .DataClass import SpectralLine


def is_list_of(my_list, expected_type):
    return isinstance(my_list, list) and all(isinstance(x, expected_type) for x in my_list)

def is_list_of_SpectralLine(data: object) -> bool:
    return isinstance(data, list) and all(isinstance(item, SpectralLine) for item in data)