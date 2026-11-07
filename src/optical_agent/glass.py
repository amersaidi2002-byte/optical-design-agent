"""SCHOTT Sellmeier dispersion. Wavelengths in micrometres.

Source: SCHOTT Optical Glass Collection datasheets (March 2018),
Constants of Dispersion Formula. See docs/physics.md for source URLs.
"""

import numpy as np

NAMES = ("N-BK7", "N-F2", "N-SF5")
B = np.array(
    [
        [1.03961212, 0.231792344, 1.01046945],
        [1.39757037, 0.159201403, 1.2686543],
        [1.52481889, 0.187085527, 1.42729015],
    ]
)
C = np.array(
    [
        [0.00600069867, 0.0200179144, 103.560653],
        [0.00995906143, 0.0546931752, 119.248346],
        [0.011254756, 0.0588995392, 129.141675],
    ]
)
WAVELENGTHS = np.array([0.4861, 0.5876, 0.6563])


def index(glass_id, wavelength):
    """Broadcast glass IDs and wavelengths; return phase refractive index."""
    ids = np.asarray(glass_id)
    if np.any((ids < 0) | (ids >= len(NAMES))) or np.any(ids != ids.astype(int)):
        raise ValueError("glass_id must contain integer IDs 0, 1, or 2")
    wave = np.asarray(wavelength)
    if np.any((wave < 0.4) | (wave > 0.7)):
        raise ValueError("catalog in this project is restricted to 0.4–0.7 µm")
    l2 = wave[..., None] ** 2
    return np.sqrt(
        1 + np.sum(B[ids.astype(int)] * l2 / (l2 - C[ids.astype(int)]), axis=-1)
    )
