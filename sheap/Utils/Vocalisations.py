"""This module contains vocalizations."""
__author__ = 'felavila'

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
