"""Primus Decision 0.1: inference runtime for the released typed-decision model (noul / choice / score).

Load the released ensemble with ``Ensemble.load("model")`` (``ensemble.py``), build a case from a request with
``request_to_case`` and answer it with ``predict_cases`` (``predict.py``); ``examples/predict_example.py`` shows the
whole path and ``python -m primus_decision.serve model`` exposes it as a JSON-lines bridge. The request and reply
shapes are specified in ``INTERFACE.md`` and ``schemas/``. Nothing here imports a transformer library.
"""

__version__ = "0.1.0"
