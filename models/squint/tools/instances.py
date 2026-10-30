"""The Squint host instances. The JavaScript release ran each of these as its own WebAssembly module; in the Gust plugin
each is compiled with its own symbol prefix (see gen_names.py) so that two of them never share state.

instance name -> source files. `verifier` is the tracker's verifier host, `acquirer` another copy of the same code that
only does full-frame acquisition (the JavaScript Acquirer owned a separate module for it)."""
VERIFIER = ['geometry.c', 'white_quads.c', 'registration.c', 'phase.c', 'qr.c', 'evidence.c', 'acquire.c', 'verify_host.c']
INSTANCES = {
    'geometry': ['geometry.c'],
    'qr': ['qr.c'],
    'quads': ['white_quads.c'],
    'flow': ['flow.c'],
    'verifier': VERIFIER,
    'acquirer': VERIFIER,
}
# Prototype groups for gen_api.py (instances with the same code share one).
API_GROUPS = {'geometry': ['geometry.c'], 'qr': ['qr.c'], 'quads': ['white_quads.c'], 'flow': ['flow.c'], 'verifier': VERIFIER}
FLAGS = ['-std=c11', '-ffp-contract=off', '-DSQUINT_FRAME_CAPACITY=921600']
