"""The Squint host instances. The JavaScript release ran each of these as its own WebAssembly module; in the Gust plugin
each is compiled with its own symbol prefix (see gen_names.py) so that two of them never share state.

instance name -> source files. `verifier` is the tracker's verifier host, `acquirer` the full-frame reader on its own (the
JavaScript Acquirer owned a whole second verifier module for it, but only ever called its acquisition)."""
VERIFIER = ['geometry.c', 'white_quads.c', 'registration.c', 'phase.c', 'qr.c', 'evidence.c', 'acquire.c', 'verify_host.c']
# The acquirer only reads whole frames: no registration, evidence certificate or tracking.
ACQUIRER = ['geometry.c', 'qr.c', 'acquire.c']
INSTANCES = {
    'geometry': ['geometry.c'],
    'qr': ['qr.c'],
    'quads': ['white_quads.c'],
    'flow': ['flow.c'],
    'verifier': VERIFIER,
    'acquirer': ACQUIRER,
}
# Prototype groups for gen_api.py: one per set of sources.
API_GROUPS = {'geometry': ['geometry.c'], 'qr': ['qr.c'], 'quads': ['white_quads.c'], 'flow': ['flow.c'], 'verifier': VERIFIER, 'acquirer': ACQUIRER}
FLAGS = ['-std=c11', '-ffp-contract=off', '-DSQUINT_FRAME_CAPACITY=921600']
