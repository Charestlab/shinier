from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
import sys
import numpy as np
from datetime import datetime
import re
import warnings

from shinier import ImageDataset, Options, ImageProcessor, REPO_ROOT
from shinier.utils import (
    Bcolors, console_log, load_images_from_folder, load_np_array, colorize,
    print_shinier_header
)

# Compute repo root as parent of /src/shinier/
IS_TTY = sys.stdin.isatty()

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

    def print_answer(answer: str = '', prefix: str = 'Selected'):
        """
        Prints the given answer with a custom prefix and updates the console output.
        Args:
            answer (str): The string representing the answer to be displayed.
            prefix (str): A custom prefix to precede the answer.
        """
        tick = ''
        if IS_TTY:
            tick = '> '
            sys.stdout.write("\033[F")  # Move cursor up one line
            sys.stdout.write("\033[K")  # Clear the line

        console_log(f"{tick}{prefix}: {answer}")
        print('')

    # Display question
    default_str = colorize('default', Bcolors.DEFAULT_TEXT)
    console_log(f"{Bcolors.COLOR_TEXT}{label} (Enter=[{Bcolors.DEFAULT_TEXT}{default}{Bcolors.COLOR_TEXT}], q=quit):{Bcolors.ENDC}")

    # Check args and set choices
    if kind == 'choice' and choices is None:
        raise ValueError('Must provide `choices` when kind == "choice"')
    if kind == 'bool':
        if choices is not None:
            warnings.warn("choices are ignored for kind 'bool'")
        if not isinstance(default, str):
            raise ValueError('`default` should be a string ("y" or "n") for kind "bool"')

    if kind == "choice" and choices:
        for i, c in enumerate(choices, 1):
            new_default_str = f" [{default_str}]" if default == i else ""
            choice_color = Bcolors.DEFAULT_TEXT if default == i else Bcolors.CHOICE_VALUE
            choice_nb = colorize(str(i), choice_color)
            console_log(f"[{choice_nb}] {c}{new_default_str}", indent_level=0, color=Bcolors.BOLD)
    if kind == 'bool':
        choices = ['Yes', 'No']
        new_default = None
        for i, c in enumerate(choices):
            is_default = c[0].lower() in default.lower()
            new_default_str = f" [{default_str}]" if is_default else ""
            if new_default is None and is_default:
                new_default = i + 1
            choice_str = f"{colorize(c[0].lower(), Bcolors.DEFAULT_TEXT)}" if default == i else colorize(c[0].lower(), Bcolors.CHOICE_VALUE)
            console_log(f"[{choice_str}] {choices[i]}{new_default_str}", indent_level=0, color=Bcolors.BOLD)
        default = new_default

    if IS_TTY:
        print("> ", end="", flush=True)
    raw = input().strip()
    # raw = input("> ").strip()

    # ---- Exit ----
    if raw.lower() == "q":
        console_log("Exit requested (q).", color=Bcolors.FAIL)
        sys.exit(0)

    # ---- Default ----
    if raw == "":
        if default is not None:
            default_ = choices[default-1] if kind in ['bool', 'choice'] else default
            print_answer(default_, prefix="Default selected")
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
            if len(val) == 0:
                raise ValueError("Expected a non-empty list of tuple. E.g.: `0, 2` or `(0, 2)` or `[0, 2]`")
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

    if kind in ['bool', 'choice']:
        prefix = 'Default selected' if val == default else 'Selected'
        print_answer(answer=choices[val-1], prefix=prefix)
    else:
        print_answer(answer=val, prefix="Answered")

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
    print_shinier_header(is_tty=IS_TTY, version= "v0.1.0")
    kwargs: Dict[str, Any] = {}

    # --------- General I/O ---------
    if images is None:
        in_dir = prompt("Input folder (directory path)?", default=str(REPO_ROOT / "INPUT"),
                        kind="str", validator=_validator_dir_exists)
        kwargs["input_folder"] = Path(in_dir).expanduser().resolve()
    else:
        kwargs["input_folder"] = None

    formats = ["png", "tif", "jpg"]
    fmt_choice = prompt("Images format?", default=1, kind="choice", choices=formats)
    kwargs["images_format"] = formats[fmt_choice - 1]

    out_dir = prompt("Output folder (directory path)?", default=str(REPO_ROOT / "OUTPUT"),
                     kind="str", validator=_validator_dir_exists)
    kwargs["output_folder"] = Path(out_dir).expanduser().resolve()

    # --------- Profile ---------
    prof = prompt("Options profile?", default=1, kind="choice", choices=["Default options", "Legacy options (will duplicate the Matlab SHINE TOOLBOX results)", "Customized options"])
    if prof == 1:
        kwargs["whole_image"] = 1

    # --------- Mask ---------
    if prof != 1:
        whole = prompt("Binary ROI masks: Analysis run on selected pixels (e.g. pixels >= 127)", default=1, kind="choice", choices=[
            "No ROI mask: Whole images will be analyzed",
            "ROI masks: Analysis run on pixels != a background pixel value you will provide",
            "ROI masks: Analysis run on pixels != most frequent pixel value in the image",
            "ROI masks: Masks loaded from the `MASK` folder and analysis run on pixels >= 127"
        ])
        kwargs["whole_image"] = whole
        if whole == 4:
            if masks is None:
                mdir = prompt("Masks folder? Will use ", default=str(REPO_ROOT / "MASK"), kind="str", validator=_validator_dir_exists)
                kwargs["masks_folder"] = Path(mdir).expanduser().resolve()
                mfmt_choice = prompt("Masks format?", default=1, kind="choice", choices=formats)
                kwargs["masks_format"] = formats[mfmt_choice - 1]
            else:
                kwargs["masks_folder"] = None
                kwargs["masks_format"] = None

        if whole in (2, 3):
            kwargs["background"] = 300 if whole == 3 else prompt("ROI masks: Analysis will be run on pixels != [input a value between 0–255]", default=127, kind="int", min_v=0, max_v=255)

    # --------- Processing Mode ---------
    mode = prompt("Processing mode", default=8, kind="choice", choices=[
        "Luminance only (lum_match)",
        "Histogram only (hist_match)",
        "Spatial frequency only (sf_match)",
        "Spectrum only (spec_match)",
        "Histogram + Spatial frequency",
        "Histogram + Spectrum",
        "Spatial frequency + Histogram",
        "Spectrum + Histogram",
        "Dithering only"
    ])
    kwargs["mode"] = mode

    # --------- Legacy Mode ---------
    if prof == 2:
        kwargs["legacy_mode"] = True

    # --------- Custom Profile ---------
    if prof == 3:
        as_gray = prompt("Load as grayscale?", default=1, kind="choice", choices=[
            "No conversion applied",
            "Equal weighted sum of R, G and B pixels is applied. (Y' = 1/3 R' + 1/3 B' + 1/3 G')",
            "Rec.ITU-R 601 is used (legacy mode; see Matlab).    (Y' = 0.299 R' + 0.587 G' + 0.114 B') (Standard-Definition monitors)",
            "Rec.ITU-R 709 is used.                              (Y' = 0.2126 R' + 0.7152 G' + 0.0722 B') (High-Definition monitors)",
            "Rec.ITU-R 2020 is used.                             (Y' = 0.2627 R' + 0.6780 G' + 0.0593 B') (Ultra-High-Definition monitors)"
        ])
        kwargs["as_gray"] = as_gray - 1
        kwargs["conserve_memory"] = prompt("Conserve memory (temp dir, 1 image in RAM)?", default='y', kind="bool")

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
        kwargs["seed"] = prompt("Provide seed for pseudo-random number generator or use time-stamped default", default=int(now.timestamp()), kind="int")

        # ---- Mode-Specific Options ----
        if mode == 1:
            kwargs["safe_lum_match"] = prompt("Safe luminance matching (will ensure pixel values fall within [0, 255])?", default='n', kind="bool")
            kwargs["target_lum"] = prompt("Target luminance list (mean, std)", default="0, 0", kind="tuple")
            rgb_weights = prompt("RGB coefficients for luminance", default=3, kind="choice", choices=[
                "Equal weights",
                "Rec.ITU-R 601 (SD monitor)",
                "Rec.ITU-R 709 (HD monitor)",
                "Rec.ITU-R 2020 (UHD monitor)"
            ])
            kwargs["rgb_weights"] = rgb_weights

        if mode in (2, 5, 6, 7, 8):
            ho = prompt("Histogram specification with SSIM optimization (see Avanaki, 2009)?", default='y', kind="bool")
            kwargs["hist_optim"] = ho != 2
            if ho == 2:
                kwargs["hist_iterations"] = prompt("How many SSIM iterations?", default=5, kind="int", min_v=1, max_v=1_000_000)
                kwargs["step_size"] = prompt("What is the SSIM step size?", default=34, kind="int", min_v=1, max_v=1_000_000)
            kwargs["hist_specification"] = None
            if not kwargs["hist_optim"]:
                hs = prompt("Which histogram specification?", default=4, kind="choice", choices=[
                    "Exact with noise (legacy)",
                    "Coltuc with moving-average filters",
                    "Coltuc with gaussian filters",
                    "Coltuc with gaussian filters and noise if residual isoluminant pixels"
                ])
                kwargs["hist_specification"] = hs - 1

            thp1 = prompt("What should be the target histogram?", default=1, kind="choice", choices=[
                'Average histogram of input images',
                'Flat histogram a.k.a. `histogram equalization`',
                'Custom: You provide one as a .npy file'
            ])
            th = [None, 'equal'][thp1-1] if thp1 in [1, 2] else 'custom'
            if th == 'custom':
                thp2 = prompt("Path to target histogram (.npy)?", kind="str")
                th = load_np_array(thp2)
            if th is not None:
                kwargs["target_hist"] = th

        if mode in (3, 4, 5, 6, 7, 8):
            rsel = prompt("What type of rescaling after sf/spec?", default=2, kind="choice",
                          choices=["none", "min/max of all images", "avg min/max"])
            kwargs["rescaling"] = rsel - 1
            if prompt("Use a specific target spectrum?", default='n', kind="bool"):
                tsp = prompt("Path to target spectrum (.npy/.txt/.csv)?", kind="str")
                ts = load_np_array(tsp)
                if ts is not None:
                    kwargs["target_spectrum"] = ts

        if mode in (5, 6, 7, 8):
            kwargs["iterations"] = prompt("How many composite iterations (hist/spec coupling)?", default=2, kind="int", min_v=1, max_v=1_000_000)

    # ---- Start SHINIER ----
    try:
        opts = Options(**kwargs)

    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}\n")

    dataset = ImageDataset(images=images, masks=masks, options=opts) if (images or masks) else ImageDataset(options=opts)
    prog_info = prompt('Select verbosity level', kind="choice", default=2, choices=[
            "None (quiet mode)",
            "Progress bar with ETA",
            "Basic progress steps (no progress bar)",
            "Detailed step-by-step info (no progress bar)",
            "Debug mode for developers (no progress bar)"
    ])
    prog_info = prog_info - 2
    ImageProcessor(dataset=dataset, verbose=prog_info, from_cli=True)

    console_log("╔══════════════════════════════════════════════════════╗")
    console_log("║                      OPTIONS                         ║")
    console_log("╚══════════════════════════════════════════════════════╝")
    # console_log("\n=== Options ===", color=Bcolors.SECTION)
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