"""The medical booking domain: doctors, slots, appointments, prescriptions.

Structurally separate from the welfare-analytics system next to it. Nothing in
this package imports the processed store, the models, the behavioral engine or
the pseudonym vault, and nothing in those imports this. See ``README.md`` in
this directory for why that separation is the design rather than an accident of
layout, and ``tests/test_medical.py`` for the test that keeps it true.
"""
