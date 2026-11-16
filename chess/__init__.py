"""ChessLabs - Chess Core Package.

Integrates local ChessLabs logic (pgn_parser, board, etc.) with the
python-chess library from site-packages.
"""

import os
import sys

_local_dir = os.path.dirname(os.path.abspath(__file__))

# Locate python-chess in site-packages
_sp_chess_dir = None
for _p in sys.path:
    if "site-packages" in _p:
        _cand = os.path.join(_p, "chess")
        if os.path.isdir(_cand) and os.path.exists(os.path.join(_cand, "__init__.py")):
            if os.path.abspath(_cand) != _local_dir:
                _sp_chess_dir = _cand
                break

if _sp_chess_dir:
    _init_file = os.path.join(_sp_chess_dir, "__init__.py")
    with open(_init_file, "r", encoding="utf-8") as _f:
        _code = compile(_f.read(), _init_file, "exec")
        exec(_code, globals())

# Ensure local ChessLabs package directory is in __path__
if "__path__" in globals() and _local_dir not in __path__:
    __path__.insert(0, _local_dir)
if _sp_chess_dir and _sp_chess_dir not in __path__:
    __path__.append(_sp_chess_dir)
