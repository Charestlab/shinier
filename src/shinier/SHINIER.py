from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import sys
from PIL import Image
from datetime import datetime

from shinier import ImageDataset, Options, ImageProcessor
from shinier.utils import Bcolors

############################################ SHINIER CLI ############################################

def SHINIER_CLI(images: Optional[np.ndarray] = None, masks: Optional[np.ndarray] = None) -> Options:
    """
    Launch the interactive CLI to build a set of SHINIER options. If images or masks are given, the CLI
    won't get the information normally recquiered to set them up (i.e., if None: ask for path and format).
    
    TERMINAL usage: Launching the SHINIER.py file directly in your terminal will launch the CLI. 
    SCRIPT usage: `from SHINIER import SHINIER_CLI` then the method SHINIER_CLI can take images or masks if desired.
    
        Args: 
            images(Optional[np.ndarray]): Images that are going to be processed.
            masks(Optional[np.ndarray]): Masks for the image processing.
    """
    console_log("\n=== SHINIER Options — Interactive CLI ===", color=Bcolors.SECTION)
    kwargs: Dict[str, Any] = {}

    # --------- General I/O ---------
    if images is None:
        in_dir = prompt_str("\nInput folder (directory path)", validator=_validator_dir_exists, default='shinier/INPUT')
        if in_dir is not None: kwargs["input_folder"] = Path(in_dir).expanduser().resolve()
    else:
        kwargs["input_folder"] = None
    
    formats = ["png", "tif", "jpg"]
    fmt_choice = prompt_choice("\nImages format", formats, default = 1)
    if fmt_choice is not None: kwargs["images_format"] = formats[fmt_choice - 1]

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
                if masks is None: 
                    mdir = prompt_str("\nMasks folder", validator=_validator_dir_exists, default='shinier/MASK')
                    if mdir is not None: kwargs["masks_folder"] = Path(mdir).expanduser().resolve()
                    
                    mfmt_choice = prompt_choice("\nMasks format", formats, default = 1)
                    if mfmt_choice is not None: kwargs["masks_format"] = formats[mfmt_choice - 1]
                else:
                    kwargs["masks_format"] = None
                    kwargs["masks_folder"] = None

            if whole in (2, 3):
                bg_auto = prompt_yes_no("\nBackground auto (most frequent luminance)?", default = True)
                if bg_auto is not None: kwargs["background"] = 300 if bg_auto else prompt_int("Background luminance É"
                ""
                "-255]", 0, 255)

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
        "Only dithering"
        ], default = 8)
    kwargs["mode"] = mode
    
    # --------- Global Preferences (default profile only) ---------    
    # Legacy
    legacy = (prof == 2)
    if legacy:
        kwargs["legacy_mode"] = True

    # --------- GLOBAL Preferences (custom profile only) ---------
    if prof == 3:
        as_gray = prompt_choice("\nLoad as grayscale?",
                                 ["No conversion applied",
                                 "Equal weighted grayscaling",
                                 "Rec. 601 (legacy)",
                                 "Rec. 709 (HD monitors)",
                                 "Rec. 2020 (UHD monitors)"
                                 ], default = 1)
        if as_gray is not None: kwargs["as_gray"] = as_gray - 1 # prompt_choice is 1-based

        conserve = prompt_yes_no("\nConserve memory (temp dir, 1 image in RAM)?", default = True)
        if conserve is not None: kwargs["conserve_memory"] = conserve

        if mode != 9:
            dith = prompt_choice("\nApply dithering before final uint8 cast?",
                                    ["No dithering", # 1
                                    "Noisy-bit dithering", # 2
                                    "Floyd-Steinberg dithering"], default = 2) # 3
        else:
            dith = prompt_choice("\nWhich dithering is going to be applied?",
                                    ["Noisy-bit dithering",       # 1 (2 for kwargs)
                                    "Floyd-Steinberg dithering"], # 2 (3 for kwargs)
                                    default = 1) 
        kwargs["dithering"] = dith - 1 if mode != 9 else dith # prompt_choice is 1-based, mode 9 values already aligned.

        now = datetime.now()
        seed = prompt_int("Random seed", default = int(now.timestamp()))
        if seed is not None: kwargs["seed"] = seed

    # --------- MODE-SPECIFIC Options (custom profile only) ---------
    if prof == 3:
        # Luminance (mode 1)
        if mode == 1:
            slm = prompt_yes_no("\nSafe luminance matching?", default = False)
            if slm is not None: kwargs["safe_lum_match"] = slm

            tl = prompt_str("\nTarget luminance list (mean, std) (comma-separated) | (0, 0) -> averages mean and std of the image set)")
            tl_list = parse_floats_list(tl)
            if tl_list is not None: kwargs["target_lum"] = tl_list
            
            rgb_weights = prompt_choice("\nWhich RGB coefficients to use for luminance conversion?",
                                    ["Equal weights", # 1
                                    "Rec.ITU-R 601 (SD monitor)", # 2
                                    "Rec.ITU-R 709 (HD monitor)", # 3
                                    "Rec.ITU-R 2020 (UHD monitor)" # 4
                                    ], default = 3) # 3
            # Doesb't substract 1 because rgb_weights is 1-base in Options.py unlike other variables
            if rgb_weights is not None: kwargs["rgb_weights"] = rgb_weights

        # Histogram-related (2,5,6,7,8)
        if mode in (2,5,6,7,8):
            hs = prompt_choice("\nHistogram specification", ["Exact (Coltuc)", "Exact with noise (legacy)"], default = 1)
            if hs is not None: kwargs["hist_specification"] = hs - 1

            ho = prompt_choice("\nSSIM optimization (Avanaki)", ["No", "Yes"], default = 1)
            if ho is not None: kwargs["hist_optim"] = ho - 1

            if ho == 2:
                iters = prompt_int("\n`SSIM iterations`", 1, 1_000_000, default = 10)
                if iters is not None: kwargs["hist_iterations"] = iters

                step = prompt_int("\nSSIM step size", 1, 1_000_000, default = 67)
                if step is not None: kwargs["step_size"] = step

            use_specific_hist = prompt_yes_no("\nUse a specific target histogram? (otherwise average will be used)", default = False)
            if use_specific_hist:
                thp = prompt_str("Path to target histogram (.npy)")
                th = load_np_array(thp)
                if th is not None: kwargs["target_hist"] = th

        # SF/Spectrum-related (3,4,5,6,7,8)
        if mode in (3,4,5,6,7,8):
            rsel = prompt_choice("\nRescaling after sf/spec", ["none", "min/max of all images", "avg min/max"], default = 2)
            if rsel is not None: kwargs["rescaling"] = rsel - 1
            use_specific_spec = prompt_yes_no("\nUse a specific target spectrum? (otherwise average will be used)", default = False)
            if use_specific_spec:
                tsp = prompt_str("\nPath to target spectrum (.npy/.txt/.csv)")
                ts = load_np_array(tsp)
                if ts is not None: kwargs["target_spectrum"] = ts
        
        # Global iterations (most useful for composite modes (hist+sf/spec or sf/hist/spec+hist)), see the README and SHINE for more details.
        if mode in (5,6,7,8):
            comp_iter = prompt_int("\nComposite iterations (hist/spec coupling)", 1, 1_000_000, default = 2)
            if comp_iter is not None: kwargs["iterations"] = comp_iter        
    #---- Starting SHINIER
    #(1) Options
    try:
        opts = Options(**kwargs)
        console_log("Starting image processing...\n", color=Bcolors.BOLD)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}\n")
    #(2) ImageDatasset with or without images/masks loaded
    if images is not None and masks is not None:
        dataset = ImageDataset(images = images, masks = masks, options = opts)
    elif images is not None:
        dataset = ImageDataset(images = images, options = opts)
    elif masks is not None :
        dataset = ImageDataset(masks = masks, options = opts)
    else:
        dataset = ImageDataset(options = opts)
    #(3) ImageProcessor with verbose 0 (minimal processing steps are printed)
    ImageProcessor(dataset = dataset, verbose = 0)
    # Preview
    console_log("\n=== Options ===", color=Bcolors.SECTION)
    for key, value in kwargs.items():
        console_log(f"{key:<20}: {value}", level=1, color=Bcolors.OKBLUE)

############################################ CLI UTILITIES ############################################

def console_log(msg: str, level: int = 0, color: Optional[str] = None):
    """Better display in the console."""
    def _set_indent_and_color(text, lev: int, col: Optional[str] = None):
        indent_str = '\t' * lev
        if col is not None:
            return "\n".join(f'{indent_str}{col}{line}{Bcolors.ENDC}' for line in text.splitlines())
        else:
            return "\n".join(f'{indent_str}{line}' for line in text.splitlines())
    print(_set_indent_and_color(msg, level, color))

def better_input(label: Optional[str], level: int = 0, color: Optional[str] = None, prompt_indicator: str = "> ") -> str:
        """ Displays the recquired input with console_log, then listens for input. """
        if label:
            console_log(label, level=level, color=color)
        return input(prompt_indicator).strip()

def prompt_str(label: str, validator=None, default=None) -> Optional[str]:
    """Prompt the user for a string input."""
    val = better_input(f"{label} (Enter = default, q = quit): ", color=Bcolors.BOLD + Bcolors.HEADER)
    if val == "" and default is not None:
        console_log(f"Default selected: {default}", level=1, color=Bcolors.OKGREEN)
        return default
    elif val == "q":
        console_log("Exit requested (q).", level=1, color=Bcolors.FAIL)
        sys.exit(0)
    if validator:
        ok, msg = validator(val)
        if not ok:
            console_log(f"✗ {msg}", level=1, color=Bcolors.FAIL)
            return prompt_str(label, validator, default)
    return val

def prompt_yes_no(label: str, default: bool) -> bool:
    """Prompt the user for a yes/no response."""
    default_str = f" [{ 'y' if default else 'n' }]"
    val = better_input(f"{label} (y/n, Enter=default{default_str}, q=quit): ", color=Bcolors.BOLD + Bcolors.HEADER).lower()
    if val == "q":
        console_log("Exit requested (q).", color=Bcolors.FAIL)
        sys.exit(0)
    if val == "":
        console_log(f"Default selected: {default}", level=1, color=Bcolors.OKGREEN)
        return default
    if val in ("y", "yes"): return True
    if val in ("n", "no"):  return False
    console_log("Please answer with y/n, Enter for default, or q to quit.", level=1, color=Bcolors.FAIL)
    return prompt_yes_no(label, default)

def prompt_int(label: str, minv: int = None, maxv: int = None, default: int = None) -> int:
    """Prompt the user for an integer input within bounds."""
    limits = []
    if minv is not None: limits.append(f"≥ {minv}")
    if maxv is not None: limits.append(f"≤ {maxv}")
    default_str = f" [{default}]" if default is not None else ""
    raw = better_input(f"{label} (Enter=default{default_str}, q=quit): ", color=Bcolors.BOLD + Bcolors.HEADER)
    if raw.lower() == "q":
        console_log("Exit requested (q).", level=1, color=Bcolors.FAIL)
        sys.exit(0)
    if raw == "":
        if default is not None:
            console_log(f"Default selected: {default}", level=1, color=Bcolors.OKGREEN)
            return default
        else:
            console_log("No default value set, please enter a value.", level=1, color=Bcolors.FAIL)
            return prompt_int(label, minv, maxv, default)
    try:
        v = int(raw)
        if minv is not None and v < minv: raise ValueError
        if maxv is not None and v > maxv: raise ValueError
        return v
    except Exception:
        console_log("Enter a valid integer " + ("(" + ", ".join(limits) + ")" if limits else ""), level=1, color=Bcolors.FAIL)
        return prompt_int(label, minv, maxv, default)

def prompt_choice(label: str, choices: List[str], default: int) -> int:
    """Prompt the user to select one option from a list."""
    console_log(label + " (Enter=default" + f" [{default}]" + ", q=quit):", color=Bcolors.BOLD + Bcolors.HEADER)
    for i, c in enumerate(choices, 1):
        console_log(f"{i}. {c}", level=1, color=Bcolors.BOLD)
    raw = better_input(None, prompt_indicator="> ")
    if raw.lower() == "q":
        console_log("Exit requested (q).", color=Bcolors.FAIL)
        sys.exit(0)
    if raw == "":
        console_log(f"Default selected: {choices[default-1]}", level=1, color=Bcolors.OKGREEN)
        return default
    try:
        k = int(raw)
        if 1 <= k <= len(choices): return k
    except Exception:
        pass
    console_log("Invalid choice.", level=1, color=Bcolors.FAIL)
    return prompt_choice(label, choices, default)

def _validator_dir_exists(v: str):
    """Validate that the directory exists."""
    if v == "":
        return (False, f"Directory must have a value.")
    else :
        p = Path(v).expanduser().resolve()
        return (p.is_dir(), f"Directory not found: {p}")

def parse_floats_list(s: Optional[str]) -> Optional[List[float]]:
    """Parse a comma/semicolon-separated list of floats into a list."""
    if not s: return None
    try:
        s = s.replace(";", ",")
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except Exception:
        console_log("Could not parse the list of numbers (use commas).", level=1, color=Bcolors.FAIL)
        return None

def load_np_array(path_str: Optional[str]) -> Optional[np.ndarray]:
    """Load a numpy array from a path if valid else return None."""
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        console_log(f"✗  File not found: {p}", level=1, color=Bcolors.FAIL)
        return None
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=False)
    console_log(f"✗  Unsupported file type: {p.suffix} (only .npy supported)", level=1, color=Bcolors.FAIL)
    return None

if __name__ == "__main__":
    SHINIER_CLI()
##########################################################################################################


####################################### For testing #######################################
def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """Load all images from a folder as numpy arrays."""
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        console_log(f"Folder not found: {folder}", color=Bcolors.FAIL)
        return []
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    arrays = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            arrays.append(np.array(img))
        except Exception as e:
            console_log(f"Failed to load {img_path}: {e}", color=Bcolors.FAIL)
    return arrays

#SHINIER_CLI(images = images_list)
