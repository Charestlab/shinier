from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
import sys
import numpy as np
from datetime import datetime
from PIL import Image
import re

from shinier import ImageDataset, Options, ImageProcessor
from shinier.utils import Bcolors, console_log, load_images_from_folder # load_np_array,

# Compute repo root as parent of /src/shinier/
REPO_ROOT = Path(__file__).resolve().parents[2]


#########################################
#            GENERIC PROMPT             #
#########################################

def prompt(
    label: str,
    default: Optional[Any] = None,
    kind: str = "str",
    choices: Optional[List[str]] = None,
    validator: Optional[Callable[[Any], Tuple[bool, str]]] = None,
    min_v: Optional[float] = None,
    max_v: Optional[float] = None,
    color: Optional[str] = None,
) -> Any:
    """Prompt user input with type casting, validation, and defaults.

    Args:
        label: Message displayed to the user.
        default: Default value returned if Enter is pressed.
        kind: Input type ('str', 'int', 'float', 'bool', 'choice').
        choices: List of valid string choices for 'choice' inputs.
        validator: Optional callable returning (ok, msg) for validation.
        min_v: Minimum numeric bound (for int/float).
        max_v: Maximum numeric bound (for int/float).
        color: Console text color.

    Returns:
        The validated and type-converted user input.
    """
    console_log(f"{label} (Enter=default [{default}], q=quit):", color=color)
    if kind == "choice" and choices:
        for i, c in enumerate(choices, 1):
            console_log(f"{i}. {c}", indent_level=1, color=Bcolors.BOLD)

    raw = input("> ").strip()

    # ---- Exit ----
    if raw.lower() == "q":
        console_log("Exit requested (q).", color=Bcolors.FAIL)
        sys.exit(0)

    # ---- Default ----
    if raw == "":
        if default is not None:
            console_log(f"Default selected: {default}", indent_level=1, color=Bcolors.OKGREEN)
            return default
        console_log("Please enter a value or specify a default.", indent_level=1, color=Bcolors.FAIL)
        return prompt(label, default, kind, choices, validator, min_v, max_v, color)

    # ---- Type conversion ----
    try:
        if kind == "int":
            val = int(raw)
            if (min_v is not None and val < min_v) or (max_v is not None and val > max_v):
                raise ValueError
        elif kind == "float":
            val = float(raw)
            if (min_v is not None and val < min_v) or (max_v is not None and val > max_v):
                raise ValueError
        elif kind == "bool":
            if raw.lower() in ("y", "yes"):
                val = True
            elif raw.lower() in ("n", "no"):
                val = False
            else:
                raise ValueError
        elif kind == "choice":
            val = int(raw)
            if not (1 <= val <= len(choices)):
                raise ValueError
        elif kind == "tuple":
            tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            val = tuple(map(float, tokens))
            if len(val) not in (2, 3):
                raise ValueError("Expected 2 or 3 float values")
        else:
            val = raw
    except ValueError:
        console_log("Invalid input.", indent_level=1, color=Bcolors.FAIL)
        return prompt(label, default, kind, choices, validator, min_v, max_v, color)

    # ---- Validation ----
    if validator:
        ok, msg = validator(val)
        if not ok:
            console_log(f"✗ {msg}", indent_level=1, color=Bcolors.FAIL)
            return prompt(label, default, kind, choices, validator, min_v, max_v, color)

    return val


#########################################
#            SHINIER CLI CORE           #
#########################################

def SHINIER_CLI(images: Optional[np.ndarray] = None, masks: Optional[np.ndarray] = None) -> Options:
    """Interactive CLI to configure and run SHINIER processing.

    Args:
        images: Optional image array to bypass folder selection.
        masks: Optional mask array to bypass folder selection.

    Returns:
        Options: Configured SHINIER options object.
    """
    console_log("\n=== SHINIER Options — Interactive CLI ===", color=Bcolors.SECTION)
    kwargs: Dict[str, Any] = {}

    # --------- General I/O ---------
    if images is None:
        in_dir = prompt("Input folder (directory path)", default=str(REPO_ROOT / "INPUT"),
                        kind="str", validator=_validator_dir_exists)
        kwargs["input_folder"] = Path(in_dir).expanduser().resolve()
    else:
        kwargs["input_folder"] = None

    formats = ["png", "tif", "jpg"]
    fmt_choice = prompt("Images format", default=1, kind="choice", choices=formats)
    kwargs["images_format"] = formats[fmt_choice - 1]

    out_dir = prompt("Output folder (directory path)", default=str(REPO_ROOT / "OUTPUT"),
                     kind="str", validator=_validator_dir_exists)
    kwargs["output_folder"] = Path(out_dir).expanduser().resolve()

    # --------- Profile ---------
    prof = prompt("Profile", default=1, kind="choice", choices=["default", "legacy", "custom"])
    if prof == 1:
        kwargs["whole_image"] = 1

    # --------- Figure/Ground ---------
    if prof != 1:
        whole = prompt("Figure-ground (whole_image)", default=1, kind="choice", choices=[
            "Whole image",
            "Figure-ground (input images as masks)",
            "Figure-ground (MASK folder)"
        ])
        kwargs["whole_image"] = whole
        if whole == 3:
            if masks is None:
                mdir = prompt("Masks folder", default=str(REPO_ROOT / "MASK"),
                              kind="str", validator=_validator_dir_exists)
                kwargs["masks_folder"] = Path(mdir).expanduser().resolve()
                mfmt_choice = prompt("Masks format", default=1, kind="choice", choices=formats)
                kwargs["masks_format"] = formats[mfmt_choice - 1]
            else:
                kwargs["masks_folder"] = None
                kwargs["masks_format"] = None

        if whole in (2, 3):
            bg_auto = prompt("Background auto (most frequent luminance)?", default=True, kind="bool")
            kwargs["background"] = 300 if bg_auto else prompt("Background luminance [0–255]",
                                                              default=128, kind="int", min_v=0, max_v=255)

    # --------- Processing Mode ---------
    mode = prompt("Processing mode", default=8, kind="choice", choices=[
        "Luminance only (lum_match)",
        "Histogram only (hist_match)",
        "Spatial frequency only (sf_match)",
        "Spectrum only (spec_match)",
        "Histogram + SF",
        "Histogram + Spectrum",
        "SF + Histogram",
        "Spectrum + Histogram",
        "Only dithering"
    ])
    kwargs["mode"] = mode

    # --------- Legacy Mode ---------
    if prof == 2:
        kwargs["legacy_mode"] = True

    # --------- Custom Profile ---------
    if prof == 3:
        as_gray = prompt("Load as grayscale?", default=1, kind="choice", choices=[
            "No conversion applied",
            "Equal weighted grayscale",
            "Rec. 601 (legacy)",
            "Rec. 709 (HD monitors)",
            "Rec. 2020 (UHD monitors)"
        ])
        kwargs["as_gray"] = as_gray - 1
        kwargs["conserve_memory"] = prompt("Conserve memory (temp dir, 1 image in RAM)?",
                                           default=True, kind="bool")

        # Dithering
        dith_choices = ["No dithering", "Noisy-bit dithering", "Floyd–Steinberg dithering"]
        if mode != 9:
            dith = prompt("Apply dithering before final uint8 cast?", default=2,
                          kind="choice", choices=dith_choices)
            kwargs["dithering"] = dith - 1
        else:
            dith = prompt("Which dithering is going to be applied?", default=1,
                          kind="choice", choices=dith_choices[1:])
            kwargs["dithering"] = dith

        # Seed
        now = datetime.now()
        kwargs["seed"] = prompt("Random seed", default=int(now.timestamp()), kind="int")

        # ---- Mode-Specific Options ----
        if mode == 1:
            kwargs["safe_lum_match"] = prompt("Safe luminance matching (will ensure pixel values fall within [0, 255]) ?", default=False, kind="bool")
            kwargs["target_lum"] = prompt("Target luminance list (mean, std)", default="0, 0", kind="tuple")
            rgb_weights = prompt("RGB coefficients for luminance", default=3, kind="choice", choices=[
                "Equal weights",
                "Rec.ITU-R 601 (SD monitor)",
                "Rec.ITU-R 709 (HD monitor)",
                "Rec.ITU-R 2020 (UHD monitor)"
            ])
            kwargs["rgb_weights"] = rgb_weights

        if mode in (2, 5, 6, 7, 8):
            hs = prompt("Histogram specification", default=4, kind="choice",
                        choices=["Exact with noise (legacy)", "Coltuc with moving-average filters", "Coltuc with gaussian filters", "Coltuc with gaussian filters and noise if residual isoluminant pixels"])
            kwargs["hist_specification"] = hs - 1
            ho = prompt("SSIM optimization (Avanaki)", default=1, kind="choice", choices=["No", "Yes"])
            kwargs["hist_optim"] = ho - 1
            if ho == 2:
                kwargs["hist_iterations"] = prompt("SSIM iterations", default=10, kind="int", min_v=1, max_v=1_000_000)
                kwargs["step_size"] = prompt("SSIM step size", default=34, kind="int", min_v=1, max_v=1_000_000)
            thp1 = prompt("What should be the target histogram?", default=1, kind="choice",
                         choices=['Average histogram of input images', 'Flat histogram a.k.a. `histogram equalization`', 'Custom: You provide one as a .npy file'])
            th = [None, 'equal'][thp1-1] if thp1 in [1, 2] else 'custom'
            if th == 'custom':
                thp2 = prompt("Path to target histogram (.npy)", kind="str")
                th = load_np_array(thp2)
            if th is not None:
                kwargs["target_hist"] = th

        if mode in (3, 4, 5, 6, 7, 8):
            rsel = prompt("Rescaling after sf/spec", default=2, kind="choice",
                          choices=["none", "min/max of all images", "avg min/max"])
            kwargs["rescaling"] = rsel - 1
            if prompt("Use a specific target spectrum?", default=False, kind="bool"):
                tsp = prompt("Path to target spectrum (.npy/.txt/.csv)", kind="str")
                ts = load_np_array(tsp)
                if ts is not None:
                    kwargs["target_spectrum"] = ts

        if mode in (5, 6, 7, 8):
            kwargs["iterations"] = prompt("Composite iterations (hist/spec coupling)", default=2,
                                          kind="int", min_v=1, max_v=1_000_000)

    # ---- Start SHINIER ----
    try:
        opts = Options(**kwargs)
        console_log("Starting image processing...\n", color=Bcolors.BOLD)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}\n")

    dataset = ImageDataset(images=images, masks=masks, options=opts) if (images or masks) else ImageDataset(options=opts)
    ImageProcessor(dataset=dataset, verbose=0)

    console_log("\n=== Options ===", color=Bcolors.SECTION)
    for key, value in kwargs.items():
        console_log(f"{key:<20}: {value}", indent_level=1, color=Bcolors.OKBLUE)


#########################################
#            UTILITIES                  #
#########################################

def _validator_dir_exists(v: str) -> Tuple[bool, str]:
    """Validate that a directory exists."""
    p = Path(v).expanduser().resolve()
    return p.is_dir(), f"Directory not found: {p}"


if __name__ == "__main__":
    SHINIER_CLI()