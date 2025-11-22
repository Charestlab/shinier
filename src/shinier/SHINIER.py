from pathlib import Path
from typing import Optional, Any, List, Callable, Tuple, get_args
import sys
import numpy as np
from datetime import datetime
import re
import warnings
from shinier import __version__ as shinier_version
from shinier.Options import ACCEPTED_IMAGE_FORMATS, OPTION_TYPES
from shinier import ImageDataset, Options, ImageProcessor, REPO_ROOT
from shinier.utils import (
    Bcolors, console_log, load_np_array, colorize,
    print_shinier_header, generate_pydantic_key_value_dict
)

# Compute repo root as parent of /src/shinier/
IS_TTY = sys.stdin.isatty()
ACCEPTED_FORMATS = list(get_args(ACCEPTED_IMAGE_FORMATS))
ESCAPE_KEYS = ['q', 'exit']


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
    console_log(f"{Bcolors.BOLD}{label} (Enter=[{Bcolors.DEFAULT_TEXT}{default}{Bcolors.ENDC}{Bcolors.BOLD}], q=quit):{Bcolors.ENDC}")

    # Check args and set choices
    if kind == 'choice' and choices is None:
        raise ValueError('Must provide `choices` when kind == "choice"')
    if kind == 'bool':
        accepted_choices = ['yes', 'no', 'y', 'n']
        if choices is not None:
            warnings.warn("choices are ignored for kind 'bool'")
        if not isinstance(default, str) or default.lower().strip() not in accepted_choices:
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
            choice_color = Bcolors.DEFAULT_TEXT if is_default else Bcolors.CHOICE_VALUE
            choice_str = f"{colorize(str(c[0].lower()), choice_color)}"
            console_log(f"[{choice_str}] {choices[i]}{new_default_str}", indent_level=0, color=Bcolors.BOLD)
        default = new_default

    if IS_TTY:
        print("> ", end="", flush=True)
    raw = input().strip()
    # raw = input("> ").strip()

    # ---- Exit ----
    if raw.lower().strip() in ESCAPE_KEYS:
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


def options_display(opts):
    """Display the options after images are processed.

    Args:
        opts (Options): Options object
    """
    types = ['io']
    if opts.whole_image != 1:
        types.append('mask')
    types += ['mode', 'color', 'dithering_memory']
    if opts.mode == 1:
        types.append('luminance')
    if opts.mode in (2, 5, 6, 7, 8):
        types.append('histogram')
    if opts.mode in (3, 4, 5, 6, 7, 8):
        types.append('fourier')
    types.append('misc')
    for option_type in types:
        console_log(f"\n---- {option_type}  ----", indent_level=1, color=Bcolors.OKBLUE)
        for key, value in dict(opts).items():
            if key in OPTION_TYPES[option_type]:
                if "target" in key:
                    if isinstance(value, np.ndarray):
                        value = "CUSTOM"
                    else:
                        value = "AVERAGE"
                console_log(f"{key:<20}: {value}", indent_level=1, color=Bcolors.OKBLUE)


#########################################
#            SHINIER CLI CORE           #
#########################################
def SHINIER_CLI(images: Optional[np.ndarray] = None, masks: Optional[np.ndarray] = None) -> ImageProcessor:
    """Interactive CLI to configure and run SHINIER processing.

    Args:
        images: Optional image array to bypass folder selection.
        masks: Optional mask array to bypass folder selection.

    Returns:
        Options: Configured SHINIER options object.
    """
    print_shinier_header(is_tty=IS_TTY, version=shinier_version)
    opts = Options()

    # --------- General I/O ---------
    if images is None:
        in_dir = prompt(f"Input folder (directory path)? Accepted formats: {ACCEPTED_FORMATS}", default=str(opts.input_folder), kind="str")
        opts.input_folder = Path(in_dir).expanduser().resolve()
        image_paths = get_image_list(opts.input_folder)
        color = Bcolors.OKGREEN if len(image_paths) > 0 else Bcolors.FAIL
        console_log(msg=f'\033[F-> {len(image_paths)} image(s) found in {opts.input_folder}\n', indent_level=0, color=color, strip=False)
    else:
        opts.input_folder = None

    out_dir = prompt("Output folder (directory path)?", default=str(opts.output_folder), kind="str")
    opts.output_folder = Path(out_dir).expanduser().resolve()

    # --------- Profile ---------
    prof = prompt("Options profile?", default=1, kind="choice", choices=["Default options", "Legacy options (will duplicate the Matlab SHINE TOOLBOX results)", "Customized options"])

    # --------- Mask ---------
    if prof != 1:
        whole = prompt("Binary ROI masks: Analysis run on selected pixels (e.g. pixels >= 127)", default=1, kind="choice", choices=[
            "No ROI mask: Whole images will be analyzed",
            "ROI masks: Analysis run on pixels != a pixel value you will provide",
            "ROI masks: Analysis run on pixels != most frequent pixel value in the image",
            "ROI masks: Masks loaded from the `MASK` folder and analysis run on pixels >= 127"
        ])

        if whole == 4:
            if masks is None:
                mdir = prompt(f"Masks folder (directory path)? Accepted formats: {ACCEPTED_FORMATS}. Will use ", default=str(REPO_ROOT / "data/MASK"), kind="str")
                opts.masks_folder = Path(mdir).expanduser().resolve()
                mask_paths = get_image_list(opts.masks_folder)
                color = Bcolors.OKGREEN if len(mask_paths) > 0 else Bcolors.FAIL
                console_log(msg=f'\033[F-> {len(mask_paths)} image(s) found in {opts.masks_folder}\n', indent_level=0, color=color, strip=False)
            else:
                opts.masks_folder = None
        opts.whole_image = [1, 2, 2, 3][whole-1]

        if whole in (2, 3):
            opts.background = opts.background if whole == 3 else prompt("ROI masks: Analysis will be run on pixels != [input a value between 0–255]", default=127, kind="int", min_v=0, max_v=255)

    # --------- Processing Mode ---------
    mode = prompt("Processing mode", default=2, kind="choice", choices=[
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
    opts.mode = mode

    # --------- Legacy Mode ---------
    if prof == 2:
        opts.legacy_mode = True

    # --------- Custom Profile ---------
    if prof == 3:
        as_gray = prompt("Load images as grayscale?", default="No", kind="bool")
        opts.as_gray = as_gray == 1
        linear_luminance = prompt("Are pixel values linearly related to luminance?", default=2, kind='choice', choices=[
            f"{Bcolors.CHOICE_VALUE}Yes [legacy mode]{Bcolors.ENDC}\n\t- No color-space conversion.\n\t- Assuming input images are linear to luminance.\n\t- All transformations will be applied independently to each channel which may produce out-of-gamut values",
            f"{Bcolors.DEFAULT_TEXT}No [default]{Bcolors.ENDC}:\n\t- Assumes input images are regular sRGB images, i.e. gamma-encoded.\n\t- Images will first be converted into CIE xyY color-space\n\t- All transformations will be applied on the luminance channel (Y) of the CIE xyY color space.\n\t- Images are then reconverted into sRGB using transformed luminance channel (Y) and original chromatic channels (x, y),\n\t- This mode should preserves color gamuts",
        ])
        opts.linear_luminance = linear_luminance == 1
        if not opts.linear_luminance:
            rec_standard = prompt("Specifies the Rec. color standard used for RGB ↔ XYZ conversion (default = 2)", default=2, kind='choice', choices=[
                f"Rec.601 (SDTV) [{Bcolors.CHOICE_VALUE}legacy mode]{Bcolors.ENDC}",
                "Rec.709 (HDTV)",
                "Rec.2020 (UHDTV, wide-gamut HDR)"
            ])
            opts.rec_standard = rec_standard

        opts.conserve_memory = prompt("Conserve memory (creates a temporary directory and keep only one image in RAM)?", default='y', kind="bool")

        # Dithering
        dith_choices = ["No dithering", "Noisy-bit dithering", "Floyd–Steinberg dithering"]
        if mode != 9:
            dith = prompt("Apply dithering before final uint8 cast?", default=1,
                          kind="choice", choices=dith_choices)
            opts.dithering = dith - 1
        else:
            dith = prompt("Which dithering is going to be applied?", default=1,
                          kind="choice", choices=dith_choices[1:])
            opts.dithering = dith

        # Seed
        now = datetime.now()
        opts.seed = prompt("Provide seed for pseudo-random number generator or use time-stamped default", default=int(now.timestamp()), kind="int")

        # ---- Mode-Specific Options ----
        if mode == 1:
            opts.safe_lum_match = prompt("Safe luminance matching (will ensure pixel values fall within [0, 255])?", default='n', kind="bool")
            opts.target_lum = prompt("Target luminance list (mean, std)", default="0, 0", kind="tuple")

        if mode in (2, 5, 6, 7, 8):
            ho = prompt("Histogram specification with SSIM optimization (see Avanaki, 2009)?", default='y', kind="bool")
            opts.hist_optim = ho != 2
            if ho == 2:
                opts.hist_iterations = prompt("How many SSIM iterations?", default=5, kind="int", min_v=1, max_v=1_000_000)
                opts.step_size = prompt("What is the SSIM step size?", default=34, kind="int", min_v=1, max_v=1_000_000)
            opts.hist_specification = None
            if not opts.hist_optim:
                hs = prompt("Which histogram specification?", default=4, kind="choice", choices=[
                    "Exact with noise (legacy)",
                    "Coltuc with moving-average filters",
                    "Coltuc with gaussian filters",
                    "Coltuc with gaussian filters and noise if residual isoluminant pixels"
                ])
                opts.hist_specification = hs - 1

            thp1 = prompt("What should be the target histogram?", default=1, kind="choice", choices=[
                'Average histogram of input images',
                'Flat histogram a.k.a. `histogram equalization`',
                'Custom: You provide one as a .npy file'
            ])
            th = [None, 'equal'][thp1-1] if thp1 in [1, 2] else 'custom'
            if th == 'custom':
                thp2 = prompt("Path to target histogram (.npy/.txt/.csv)?", kind="str")
                th = load_np_array(thp2)
            if th is not None:
                opts.target_hist = th

        if mode in (3, 4, 5, 6, 7, 8):
            rsel = prompt("What type of rescaling after sf/spec?", default=2, kind="choice",
                          choices=["none", "min/max of all images", "avg min/max"])
            opts.rescaling = rsel - 1
            ans = prompt("Use a specific target spectrum?", default='n', kind="bool")
            if ans == 1:
                tsp = prompt("Path to target spectrum (.npy/.txt/.csv)?", kind="str")
                ts = load_np_array(tsp)
                if ts is not None:
                    opts.target_spectrum = ts

        if mode in (5, 6, 7, 8):
            opts.iterations = prompt("How many composite iterations (hist/spec coupling)?", default=2, kind="int", min_v=1, max_v=1_000_000)

    # ---- Progress info ----
    if prof != 1:
        prog_info = prompt('Select verbosity level', kind="choice", default=2, choices=[
                "None (quiet mode)",
                "Progress bar with ETA",
                "Basic progress steps (no progress bar)",
                "Detailed step-by-step info (no progress bar)",
                "Debug mode for developers (no progress bar)"
        ])
        opts.verbose = prog_info - 2

    # ---- Start SHINIER ----
    dataset = ImageDataset(images=images, masks=masks, options=opts) if (images or masks) else ImageDataset(options=opts)
    results = ImageProcessor(dataset=dataset, verbose=opts.verbose, from_cli=True)

    console_log("╔══════════════════════════════════════════════════════╗")
    console_log("║                      OPTIONS                         ║")
    console_log("╚══════════════════════════════════════════════════════╝")

    options_display(opts)

    return results


def get_image_list(src_path: Path):

    def is_image(file: Path):
        return file.is_file() and len(file.suffix) > 1 and file.suffix.lower()[1:] in ACCEPTED_FORMATS

    # Convert to Path if input_data is a string
    input_path = Path(src_path) if isinstance(src_path, str) else src_path

    # Handle cases with wildcards
    if "*" in str(input_path):  # Check if it's a glob pattern
        directory = input_path.parent
        pattern = input_path.name
        all_files = sorted(directory.glob(pattern))
    else:
        all_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

    # Filter to include only recognized image files
    return sorted([p for p in all_files if is_image(p)])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SHINIER")
    parser.add_argument("--show_results", action="store_true",
                        help="Display (or save) a summary figure showing before/after processing.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Save the overview figure instead of displaying it. Provide output path, e.g. figure.png")
    parser.add_argument("--image_index", type=int, default=0,
                        help="Image index for the overview figure (default: 0)")

    args = parser.parse_args()

    # Run the interactive core
    processor = SHINIER_CLI()

    # Overview mode
    if args.show_results:
        from shinier.utils import show_processing_overview
        try :
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError(
                "Matplotlib is not installed. "
                "Install with: pip install shinier[viz]"
            )
        fig = show_processing_overview(processor, img_idx=args.image_index, show_figure=False)

        if args.save_path is not None:
            fig.savefig(args.save_path, dpi=150)
            print(f"Processing overview figure saved successfully at: {args.save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
