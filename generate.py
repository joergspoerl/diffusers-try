#!/usr/bin/env python3
"""Konsolidierter Wrapper: leitet alle Aufrufe auf generate_mod.py um.

Beibehaltung des alten Dateinamens 'generate.py' für Rückwärtskompatibilität.
Der komplette Funktionsumfang befindet sich jetzt in generate_mod.py.
"""
from generate_mod import main

if __name__ == "__main__":  # pragma: no cover
    main()
