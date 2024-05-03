"""
main.py
core inference routines
"""

from lm import LanguageModel

model = LanguageModel()
model("""
anon1: What?
""".strip())


