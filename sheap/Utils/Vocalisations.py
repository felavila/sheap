"""This module contains vocalizations."""
__version__ = '0.1.0'
__author__ = 'Felipe Avila-Vera'
# Auto-generated __all__
__all__ = [
    "sounds",
]

def sounds(language):
    if language.lower() == "spanish":
        print("MEEE")
    elif language.lower() == "english":
        print("Baa Baa")
    else:
        print("I dont know how sheeps sounds in that language")
