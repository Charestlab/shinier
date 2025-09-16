#!/usr/bin/env python3
# cli_options.py

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import sys
from shinier import ImageDataset, Options, ImageProcessor
from shinier.utils import compute_metrics_from_images
from PIL import Image

def prompt_str(label: str, validator=None, default=None) -> Optional[str]:
    """Prompt the user for a string input.

    Args:
        label (str): The message displayed to the user.
        validator (Optional[callable]): A function that takes a string input and 
            returns a tuple (is_valid: bool, message: str). Used to validate the input. 
            Defaults to None.
        default (Optional[str]): Value that will be selected if defined and Enter is pressed.

    Returns:
        Optional[str]: The validated string input.
    """
    val = input(f"{label} (Enter = default, q = quit): ").strip()
    if val == "" and default is not None:
        print(f"    Default selected: {default}")
        return default
    elif val == "q":
        print("    Exit requested (q).")
        sys.exit(0)
    if validator:
        ok, msg = validator(val)
        if not ok:
            print(f"  ✗ {msg}")
            return prompt_str(label, validator)
    return val

def prompt_yes_no(label: str, default: bool) -> bool:
    """Prompt the user for a yes/no response, with default.

    Args:
        label (str): The message displayed to the user.
        default (bool): Default value if Enter is pressed.

    Returns:
        bool: True for yes, False for no.
    """
    default_str = f" [{ 'y' if default else 'n' }]"
    val = input(f"{label} (y/n, Enter=default{default_str}, q=quit): ").strip().lower()
    if val == "q":
        print("Exit requested (q).")
        sys.exit(0)
    if val == "":
        print(f"    Default selected: {default}")
        return default
    if val in ("y", "yes"): return True
    if val in ("n", "no"):  return False
    print("  Please answer with y/n, Enter for default, or q to quit.")
    return prompt_yes_no(label, default)

def prompt_int(label: str, minv: int = None, maxv: int = None, default: int = None) -> int:
    """Prompt the user for an integer input within bounds, with default.

    Args:
        label (str): The message displayed to the user.
        minv (int): Minimum allowed integer.
        maxv (int): Maximum allowed integer.
        default (int): Default value if Enter is pressed.

    Returns:
        int: The validated integer input.
    """
    limits = []
    if minv is not None: limits.append(f"≥ {minv}")
    if maxv is not None: limits.append(f"≤ {maxv}")
    default_str = f" [{default}]" if default is not None else ""
    raw = input(f"{label} (Enter=default{default_str}, q=quit): ").strip()
    if raw.lower() == "q":
        print("Exit requested (q).")
        sys.exit(0)
    if raw == "":
        if default is not None:
            print(f"   Default selected: {default}")
            return default
        else:
            print("   No default value set, please enter a value.")
            return prompt_int(label, minv, maxv, default)
    try:
        v = int(raw)
        if minv is not None and v < minv: raise ValueError
        if maxv is not None and v > maxv: raise ValueError
        return v
    except Exception:
        print("  Enter a valid integer " + ("(" + ", ".join(limits) + ")" if limits else ""))
        return prompt_int(label, minv, maxv, default)

def prompt_choice(label: str, choices: List[str], default: int) -> int:
    """Prompt the user to select one option from a list, with default.

    Args:
        label (str): The message displayed to the user.
        choices (List[str]): The list of choices.
        default (int): Default choice index (1-based) if Enter is pressed.

    Returns:
        int: The chosen option index (1-based).
    """
    print(label + " (Enter=default" + f" [{default}]" + ", q=quit):")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    raw = input("Choice #: ").strip()
    if raw.lower() == "q":
        print("Exit requested (q).")
        sys.exit(0)
    if raw == "":
        print(f"  Default selected: {choices[default-1]}")
        return default
    try:
        k = int(raw)
        if 1 <= k <= len(choices): return k
    except Exception:
        pass
    print("  Invalid choice.")
    return prompt_choice(label, choices, default)

def _validator_image_fmt(v: str):
    """Validate that the image format is supported."""
    allowed = {"png","tif","tiff","jpg","jpeg"}
    return (v.lower() in allowed, f"Format must be one of {sorted(allowed)}")

def _validator_dir_exists(v: str):
    """Validate that the directory exists."""
    if v == "":
        return (False, f"Directory must have a value.")
    else :
        p = Path(v).expanduser().resolve()
        return (p.is_dir(), f"Directory not found: {p}")

def _validator_mask_fmt(v: str):
    """Validate that the mask format is supported."""
    return _validator_image_fmt(v)

def parse_floats_list_or_none(s: Optional[str]) -> Optional[List[float]]:
    """Parse a comma/semicolon-separated list of floats.

    Args:
        s (Optional[str]): String of comma-separated floats.

    Returns:
        Optional[List[float]]: List of floats or None if invalid.
    """
    if not s: return None
    try:
        s = s.replace(";", ",")
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except Exception:
        print("  Could not parse the list of numbers (use commas).")
        return None

def load_array_maybe(path_str: Optional[str]) -> Optional[np.ndarray]:
    """Load a numpy array from a path if valid.

    Args:
        path_str (Optional[str]): Path to a numpy array file.

    Returns:
        Optional[np.ndarray]: Loaded numpy array or None.
    """
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        print(f"  ⚠️  File not found: {p}")
        return None
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=False)
    print(f"  ⚠️  Unsupported file type: {p.suffix} (only .npy supported)")
    return None

############################################ SHINIER CLI ############################################

def SHINIER_CLI(images = None, masks = None) -> Options:
    """Launch the interactive CLI to build a set of SHINIER options."""
    print("\n=== SHINIER Options — Interactive CLI ===")
    print("(q = immediate quit)\n")

    kwargs: Dict[str, Any] = {}

    # --------- General I/O ---------
    # If images are provided, doesn't ask for path or format
    if images is None:
        in_dir = prompt_str("\nInput folder (directory path)", validator=_validator_dir_exists, default='shinier/INPUT')
        if in_dir is not None: 
            kwargs["input_folder"] = Path(in_dir).expanduser().resolve()
    else:
        kwargs["input_folder"] = None
    
    fmt = prompt_str("\nImages format [png/tif/tiff/jpg/jpeg]", validator=_validator_image_fmt, default="tif")
    if fmt is not None: 
        kwargs["images_format"] = fmt

    out_dir = prompt_str("\nOutput folder (directory path)", validator=_validator_dir_exists, default='shinier/OUTPUT')
    if out_dir is not None: kwargs["output_folder"] = Path(out_dir).expanduser().resolve()

    # --------- Profile ---------
    prof = prompt_choice("\nProfile", ["default", "legacy", "custom"], default = 1)
    
    if prof == 1:
        kwargs["whole_image"] = 1 # Whole image

    # --------- Figure/Ground ---------
    if prof != 1:
        whole = prompt_choice("\nFigure-ground (whole_image)", [
            "Whole image",
            "Figure-ground (input images as masks)",
            "Figure-ground (MASK folder)"
        ], default = 1)
        if whole is not None:
            kwargs["whole_image"] = whole
            if whole == 3:
                # If masks are provided, doesn't ask for path or format
                if masks is None:
                    mdir = prompt_str("\nMasks folder", validator=_validator_dir_exists, default='shinier/MASK')
                    if mdir is not None: 
                        kwargs["masks_folder"] = Path(mdir).expanduser().resolve()
                    mfmt = prompt_str("\nMasks format [png/tif/tiff/jpg/jpeg]", validator=_validator_mask_fmt, default="tif")
                    if mfmt is not None: 
                        kwargs["masks_format"] = mfmt
                else:
                    kwargs["masks_format"] = None
                    kwargs["masks_folder"] = None

            if whole in (2, 3):
                bg_auto = prompt_yes_no("\nBackground auto (most frequent luminance)?", default = True)
                if bg_auto is not None: 
                    kwargs["background"] = 300 if bg_auto else prompt_int("Background luminance [0-255]", 0, 255)

    # --------- Processing Mode ---------
    mode = prompt_choice("\nProcessing mode", [
        "Luminance only (lummatch)",
        "Histogram only (hist_match)",
        "Spatial frequency only (sf_match)",
        "Spectrum only (spec_match)",
        "Histogram + SF",
        "Histogram + Spectrum",
        "SF + Histogram",
        "Spectrum + Histogram",
        "Noisy bit-dithering"
    ], default = 8)
    kwargs["mode"] = mode
    
    # --------- Global Preferences (default profile only) ---------
    # Default
    if prof == 1 :
        print(mode)
        kwargs["rescaling"] = 0 if mode in [1,2] else 1
    
    # Legacy
    legacy = (prof == 2)
    if legacy:
        kwargs["legacy_mode"] = True

    # --------- Global Preferences (custom profile only) ---------
    if prof == 3:
        as_gray = prompt_yes_no("\nLoad as grayscale?", default = False)
        if as_gray is not None: 
            kwargs["as_gray"] = as_gray

        conserve = prompt_yes_no("\nConserve memory (temp dir, 1 image in RAM)?", default = True)
        if conserve is not None: 
            kwargs["conserve_memory"] = conserve
        
        dith = prompt_yes_no("\nApply dithering before final uint8 cast?", default = True) if mode != 9 else True
        if dith is not None: 
            kwargs["dithering"] = dith

        seed = prompt_int("Random seed")
        if seed is not None: 
            kwargs["seed"] = seed

        # Metrics
        msel = prompt_choice("\nMetrics (empty = Options defaults)", ["rmse", "ssim", "none", "rmse+ssim"], default = 4)
        if msel is not None:
            if msel == 1: kwargs["metrics"] = ["rmse"]
            elif msel == 2: kwargs["metrics"] = ["ssim"]
            elif msel == 3: kwargs["metrics"] = []
            else: kwargs["metrics"] = ["rmse","ssim"]

    # --------- Mode-specific Options (custom profile only) ---------
    if prof == 3:
        # Luminance (mode 1)
        if mode == 1:
            slm = prompt_yes_no("\nSafe luminance matching?", default = False)
            if slm is not None: 
                kwargs["safe_lum_match"] = slm
            tl = prompt_str("\nTarget luminance list (mean, std) (comma-separated)\n\t (0, 0) -> averages mean and std of the image set)")
            tl_list = parse_floats_list_or_none(tl)
            if tl_list is not None: 
                kwargs["target_lum"] = tl_list

        # Histogram-related (2,5,6,7,8)
        if mode in (2,5,6,7,8):
            hs = prompt_choice("\nHistogram specification", ["Exact (Coltuc)", "Exact with noise (legacy)"], default = 1)
            if hs is not None: 
                kwargs["hist_specification"] = hs - 1

            ho = prompt_choice("\nSSIM optimization (Avanaki)", ["No", "Yes"], default = 1)
            if ho is not None: 
                kwargs["hist_optim"] = ho - 1

            if ho == 2:
                iters = prompt_int("\nSSIM iterations", 1, 1_000_000)
                if iters is not None: 
                    kwargs["iterations"] = iters
                step = prompt_int("\nSSIM step size", 1, 1_000_000)
                if step is not None: 
                    kwargs["step_size"] = step

            use_specific_hist = prompt_yes_no("\nUse a specific target histogram? (otherwise average will be used)", default = False)
            if use_specific_hist:
                thp = prompt_str("Path to target histogram (.npy)")
                th = load_array_maybe(thp)
                if th is not None:
                    kwargs["target_hist"] = th

        # SF/Spectrum-related (3,4,5,6,7,8)
        if mode in (3,4,5,6,7,8):
            rsel = prompt_choice("\nRescaling after sf/spec", ["none", "min/max of all images", "avg min/max"], default = 2)
            if rsel is not None: 
                kwargs["rescaling"] = rsel - 1
            use_specific_spec = prompt_yes_no("\nUse a specific target spectrum? (otherwise average will be used)", default = False)
            if use_specific_spec:
                tsp = prompt_str("\nPath to target spectrum (.npy/.txt/.csv)")
                ts = load_array_maybe(tsp)
                if ts is not None:
                    kwargs["target_spectrum"] = ts
        
        kwargs["rescaling"] = 0 if mode in [1,2] else 1
   
    # Build Options object
    try:
        opts = Options(**kwargs)
    except Exception as e:
        print(f"\n Invalid configuration: {e}\n", file=sys.stderr)
        raise

    if (images and masks) is not None:
        dataset = ImageDataset(images = images, masks = masks, options = opts)
    elif images is not None:
        dataset = ImageDataset(images = images, options = opts)
    elif masks is not None :
        dataset = ImageDataset(masks = masks, options = opts)
    else:
        dataset = ImageDataset(options = opts)

    ok = ImageProcessor(dataset=dataset, verbose=True)

    # Preview
    print("=== Options ===")
    print(opts)

def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """Load all images from a folder as numpy arrays."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        return []
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    arrays = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            arrays.append(np.array(img))
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    return arrays

# Example usage:
images_list = load_images_from_folder("/Users/mathiassalvas/projects/shinier-toolbox/testing_INPUT")

#SHINIER_CLI(images = images_list)
SHINIER_CLI()