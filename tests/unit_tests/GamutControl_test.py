# tests/unit_tests/GamutControl_test.py

import numpy as np
import pytest

from shinier.color import ColorConverter, GamutControl

pytestmark = pytest.mark.unit_tests


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def converter():
    return ColorConverter(rec_standard="rec709")


@pytest.fixture
def gamut(converter):
    return GamutControl(_converter=converter)


def random_xyY(H=64, W=64):
    x = np.clip(0.3127 + 0.1*(np.random.rand(H, W)-0.5), 0.05, 0.75)
    y = np.clip(0.3290 + 0.1*(np.random.rand(H, W)-0.5), 0.05, 0.85)
    Y = np.random.rand(H,W)
    return x, y, Y


# ---------------------------------------------------------
# 1. Neutral gray construction
# ---------------------------------------------------------

def test_neutral_gray_is_achromatic(gamut):

    Y = np.random.rand(32, 32)
    gray = gamut._neutral_rgb_for_Y(Y)

    # Neutral should be achromatic in linear RGB. Allow tiny numerical error.
    max_rg = float(np.max(np.abs(gray[..., 0] - gray[..., 1])))
    max_gb = float(np.max(np.abs(gray[..., 1] - gray[..., 2])))
    assert max_rg < 3e-4
    assert max_gb < 3e-4


# ---------------------------------------------------------
# 2. Luminance constraint never increases Y
# ---------------------------------------------------------

def test_luminance_constraint_never_increases_Y(gamut):

    x, y, Y = random_xyY()

    Y2, _ = gamut._apply_constrain_image_luminance(Y, np.dstack([x,y]))

    assert np.all(Y2 <= Y + 1e-12)


# ---------------------------------------------------------
# 3. Chrominance constraint preserves luminance
# ---------------------------------------------------------

def test_chroma_constraint_preserves_luminance(gamut):

    x, y, Y = random_xyY()

    Y2, _ = gamut._apply_constrain_image_chrominance(Y, np.dstack([x,y]))

    assert np.allclose(Y, Y2)


# ---------------------------------------------------------
# 4. Chrominance never increases distance from neutral
# ---------------------------------------------------------

def test_chroma_never_increases_distance(converter, gamut):

    x, y, Y = random_xyY()

    xyz = converter.xyY_to_xyz(np.dstack([x,y,Y]))
    rgb_before = converter.xyz_to_linRGB(xyz)

    Y2, xy2 = gamut._apply_constrain_image_chrominance(Y, np.dstack([x,y]))

    xyz2 = converter.xyY_to_xyz(np.dstack([xy2[...,0],xy2[...,1],Y2]))
    rgb_after = converter.xyz_to_linRGB(xyz2)

    gray = gamut._neutral_rgb_for_Y(Y)

    d_before = np.linalg.norm(rgb_before-gray,axis=-1)
    d_after  = np.linalg.norm(rgb_after-gray,axis=-1)

    assert np.all(d_after <= d_before + 1e-12)


# ---------------------------------------------------------
# 5. Dataset luminance uses global factor
# ---------------------------------------------------------

def test_dataset_luminance_scaling_is_global(gamut):

    x, y, Y = random_xyY()

    Y2, _ = gamut._apply_constrain_dataset_luminance(
        [Y.copy(), Y.copy()],
        [np.dstack([x,y]), np.dstack([x,y])]
    )

    ratio0 = Y2[0]/Y
    ratio1 = Y2[1]/Y

    assert np.allclose(ratio0, ratio1)


# ---------------------------------------------------------
# 6. Image luminance scaling differs per image
# ---------------------------------------------------------

def test_image_luminance_scaling_is_per_image(gamut):

    # We want to guarantee that, for some chromaticity (x,y), a high-Y image
    # produces out-of-gamut linear RGB values (so compression is required), while
    # a low-Y image remains in-gamut.
    #
    # The exact (x,y) that triggers overflow depends on the RGB<->XYZ matrices,
    # so we search deterministically.

    H, W = 32, 32

    def makes_overflow(x: float, y: float, Y01: float) -> bool:
        x_img = np.full((H, W), x, dtype=np.float64)
        y_img = np.full((H, W), y, dtype=np.float64)
        Y_img = np.full((H, W), Y01, dtype=np.float64)
        xyY = np.dstack([x_img, y_img, Y_img])
        xyz = gamut._converter_raw.xyY_to_xyz(xyY)
        rgb = gamut._converter_raw.xyz_to_linRGB(xyz)
        return bool(np.any(rgb > 1.0 + 1e-9))

    # Search space: avoid y~0 singularities and ensure x+y<1.
    xs = np.linspace(0.05, 0.80, 24)
    ys = np.linspace(0.05, 0.85, 24)
    Y_hi = 0.98
    Y_lo = 0.10

    found = False
    x0 = 0.0
    y0 = 0.0
    for x in xs:
        for y in ys:
            if x + y >= 0.98:
                continue
            # We want a chromaticity that forces overflow at high luminance but NOT at low luminance.
            if makes_overflow(float(x), float(y), Y01=Y_hi) and (not makes_overflow(float(x), float(y), Y01=Y_lo)):
                x0, y0 = float(x), float(y)
                found = True
                break
        if found:
            break

    assert found, "Could not find an (x,y) that overflows at high Y but not at low Y under current matrices."

    other = np.dstack([
        np.full((H, W), x0, dtype=np.float64),
        np.full((H, W), y0, dtype=np.float64),
    ])

    # High-Y should be compressed.
    Y_hi_img = np.full((H, W), Y_hi * 255, dtype=np.float64)
    # Low-Y should remain unchanged.
    Y_lo_img = np.full((H, W), Y_lo * 255, dtype=np.float64)

    Y_hi_c, _ = gamut._apply_constrain_image_luminance(Y_hi_img, other)
    Y_lo_c, _ = gamut._apply_constrain_image_luminance(Y_lo_img, other)

    r_hi = float(np.mean(Y_hi_c / Y_hi_img))
    r_lo = float(np.mean(Y_lo_c / Y_lo_img))

    # High-Y must be compressed.
    assert r_hi < 0.999
    # Low-Y should not need compression for the selected (x,y).
    assert np.isclose(r_lo, 1.0)


# ---------------------------------------------------------
# 7. Already in-gamut images remain unchanged
# ---------------------------------------------------------

def test_no_change_if_already_ingamut(gamut):

    x, y, Y = random_xyY()

    Y_small = 0.2*Y

    Y2, xy2 = gamut._apply_constrain_image_luminance(
        Y_small,
        np.dstack([x,y])
    )

    assert np.allclose(Y_small,Y2)
    assert np.allclose(x,xy2[...,0])
    assert np.allclose(y,xy2[...,1])


# ---------------------------------------------------------
# 8. Stability near black (Y≈0)
# ---------------------------------------------------------

def test_low_luminance_is_stable(gamut):

    x, y, Y = random_xyY()

    Y *= 1e-6

    Y2,_ = gamut._apply_constrain_image_luminance(Y,np.dstack([x,y]))

    assert np.isfinite(Y2).all()
