#!/usr/bin/env bash
set -euo pipefail

# Compare SHINIER (Python) against the original MATLAB SHINE toolbox.
# Requires MATLAB and the SHINE toolbox:
#   http://www.mapageweb.umontreal.ca/gosselif/SHINE/
#
# Centralized paths and run settings. Override any of these from the shell, e.g.
# RUN_ROOT=tmp/matlab_shine_comparison/my_run LIMIT=8 bash tests/tools/run_matlab_shine_comparison.sh
# If MATLAB/Python/SHINE are not found, the script asks for the needed paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MATLAB_BIN="${MATLAB_BIN:-}"
SHINE_DIR="${SHINE_DIR:-}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/tmp/matlab_shine_comparison}"
RUN_NAME="${RUN_NAME:-full_comparison}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_DIR/$RUN_NAME}"
MATLAB_VS_PYTHON_ROOT="$RUN_ROOT/matlab_vs_python"
PYTHON_TARGET_ROOT="$RUN_ROOT/python_targets"

MODES="${MODES:-1 2 3 4 5 6 7 8}"
LIMIT="${LIMIT:-8}"
ITERATIONS="${ITERATIONS:-5}"
SEED="${SEED:-42}"
FULL_TRACKING="${FULL_TRACKING:-0}"

PY_SCRIPT="$SCRIPT_DIR/matlab_shine_comparison.py"
REQUIRED_SHINE_FILES=(lumMatch.m histMatch.m sfMatch.m specMatch.m)
REQUIRED_SHINE_FUNCTIONS=(lumMatch histMatch sfMatch specMatch)

read -r -a MODE_ARGS <<< "$MODES"

BOLD="$(printf '\033[1m')"
RESET="$(printf '\033[0m')"

is_executable() {
    local candidate="$1"
    command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]
}

detect_matlab_bin() {
    local requested="$1"
    local candidate
    local found=""

    if [[ -n "$requested" && "$requested" != "matlab" && -x "$requested" ]]; then
        printf '%s' "$requested"
        return
    fi

    if command -v matlab >/dev/null 2>&1; then
        command -v matlab
        return
    fi

    for candidate in \
        /Applications/MATLAB_R*.app/bin/matlab \
        /Applications/MATLAB.app/bin/matlab \
        /usr/local/MATLAB/R*/bin/matlab \
        /opt/MATLAB/R*/bin/matlab \
        /usr/local/bin/matlab \
        /usr/bin/matlab \
        "/c/Program Files/MATLAB"/R*/bin/matlab.exe \
        "/mnt/c/Program Files/MATLAB"/R*/bin/matlab.exe
    do
        if [[ -x "$candidate" ]]; then
            found="$candidate"
        fi
    done

    printf '%s' "$found"
}

print_path_help() {
    local label="$1"

    case "$label" in
        Python)
            printf '\n%sPython executable path%s\n' "$BOLD" "$RESET" >&2
            printf '  What to provide: the Python executable used to run this diagnostic.\n' >&2
            printf '  Examples:\n' >&2
            printf '    - python3\n' >&2
            printf '    - .venv/bin/python\n' >&2
            printf '    - /Users/you/environments/shinier_venv/bin/python\n' >&2
            ;;
        MATLAB)
            printf '\n%sMATLAB executable path%s\n' "$BOLD" "$RESET" >&2
            printf '  What to provide: the MATLAB executable, not the MATLAB.app folder.\n' >&2
            printf '  Examples:\n' >&2
            printf '    - /Applications/MATLAB_R2025a.app/bin/matlab\n' >&2
            printf '    - /usr/local/MATLAB/R2025a/bin/matlab\n' >&2
            printf '    - C:\\Program Files\\MATLAB\\R2025a\\bin\\matlab.exe\n' >&2
            ;;
        SHINE)
            printf '\n%sSHINE toolbox folder%s\n' "$BOLD" "$RESET" >&2
            printf '  What to provide: the folder containing MATLAB files such as:\n' >&2
            printf '    - lumMatch.m\n' >&2
            printf '    - histMatch.m\n' >&2
            printf '    - sfMatch.m\n' >&2
            printf '    - specMatch.m\n' >&2
            printf '  Example:\n' >&2
            printf '    - /Users/you/projects/shinetoolbox\n' >&2
            printf '  Download SHINE:\n' >&2
            printf '    - http://www.mapageweb.umontreal.ca/gosselif/SHINE/\n' >&2
            printf '  Press Enter only if MATLAB can already find those SHINE functions.\n' >&2
            ;;
    esac
}

prompt_required_path() {
    local label="$1"
    local current="$2"
    local env_var="$3"
    local value="$current"

    while ! is_executable "$value"; do
        printf '%s not found: %s\n' "$label" "$value" >&2
        if [[ ! -t 0 ]]; then
            print_path_help "$label"
            printf 'Set %s=/path/to/executable or run this script interactively.\n' "$env_var" >&2
            exit 1
        fi
        print_path_help "$label"
        printf '\nEnter %s executable path: ' "$label" >&2
        read -r value
        if [[ -z "$value" ]]; then
            printf '%s path is required.\n' "$label" >&2
            exit 1
        fi
    done

    printf '%s' "$value"
}

prompt_optional_dir() {
    local label="$1"
    local current="$2"
    local env_var="$3"
    local value="$current"

    while [[ -n "$value" && ! -d "$value" ]]; do
        printf '%s directory not found: %s\n' "$label" "$value" >&2
        if [[ ! -t 0 ]]; then
            print_path_help "$label"
            printf 'Set %s=/path/to/directory or leave it unset if MATLAB already has it on path.\n' "$env_var" >&2
            exit 1
        fi
        print_path_help "$label"
        printf '\nEnter %s directory path, or press Enter if MATLAB already has it on path: ' "$label" >&2
        read -r value
    done

    if [[ -z "$value" && -t 0 && -z "${SHINE_DIR:-}" ]]; then
        print_path_help "$label"
        printf '\nEnter %s directory path, or press Enter if MATLAB already has it on path: ' "$label" >&2
        read -r value
        while [[ -n "$value" && ! -d "$value" ]]; do
            printf '%s directory not found: %s\n' "$label" "$value" >&2
            print_path_help "$label"
            printf '\nEnter %s directory path, or press Enter if MATLAB already has it on path: ' "$label" >&2
            read -r value
        done
    fi

    printf '%s' "$value"
}

validate_shine_dir() {
    local missing=()
    local file

    for file in "${REQUIRED_SHINE_FILES[@]}"; do
        if [[ ! -f "$SHINE_DIR/$file" ]]; then
            missing+=("$file")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        printf '\n%sSHINE toolbox check failed%s\n' "$BOLD" "$RESET" >&2
        printf '  Folder checked:\n' >&2
        printf '    - %s\n' "$SHINE_DIR" >&2
        printf '  Missing required MATLAB files:\n' >&2
        for file in "${missing[@]}"; do
            printf '    - %s\n' "$file" >&2
        done
        printf '  Download SHINE:\n' >&2
        printf '    - http://www.mapageweb.umontreal.ca/gosselif/SHINE/\n' >&2
        printf '  Then rerun with SHINE_DIR=/path/to/shinetoolbox.\n' >&2
        exit 1
    fi
}

validate_shine_matlab_path() {
    local matlab_code
    local function_list

    function_list="$(printf "'%s'," "${REQUIRED_SHINE_FUNCTIONS[@]}")"
    function_list="${function_list%,}"
    matlab_code="funcs={${function_list}}; missing={}; for k=1:numel(funcs), if exist(funcs{k}, 'file') ~= 2, missing{end+1}=funcs{k}; end; end; if ~isempty(missing), fprintf(2, '\\nSHINE toolbox check failed\\n'); fprintf(2, '  MATLAB cannot find required SHINE functions:\\n'); for k=1:numel(missing), fprintf(2, '    - %s\\n', missing{k}); end; fprintf(2, '  Provide SHINE_DIR=/path/to/shinetoolbox or download SHINE from:\\n'); fprintf(2, '    - http://www.mapageweb.umontreal.ca/gosselif/SHINE/\\n'); exit(1); end"

    if ! "$MATLAB_BIN" -batch "$matlab_code"; then
        printf '\n%sSHINE toolbox check failed%s\n' "$BOLD" "$RESET" >&2
        printf '  MATLAB could not confirm that SHINE is on the MATLAB path.\n' >&2
        printf '  Provide SHINE_DIR=/path/to/shinetoolbox and rerun.\n' >&2
        exit 1
    fi
}

validate_shine_setup() {
    printf '\n[preflight] Checking SHINE toolbox availability\n'
    if [[ -n "$SHINE_DIR" ]]; then
        validate_shine_dir
    else
        validate_shine_matlab_path
    fi
}

run_matlab_runner() {
    local runner="$1"
    local matlab_runner="${runner//\'/\'\'}"
    "$MATLAB_BIN" -batch "run('$matlab_runner')"
}

cleanup_csv_only_root() {
    local root="$1"
    local path

    [[ -d "$root" ]] || return 0
    find "$root" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
    while IFS= read -r -d '' path; do
        case "$path" in
            *.csv) ;;
            *) rm -f "$path" ;;
        esac
    done < <(find "$root" -mindepth 1 -maxdepth 1 -type f -print0)
}

cleanup_default_artifacts() {
    if [[ "$FULL_TRACKING" == "1" || "$FULL_TRACKING" == "true" || "$FULL_TRACKING" == "yes" ]]; then
        return 0
    fi
    cleanup_csv_only_root "$MATLAB_VS_PYTHON_ROOT"
    cleanup_csv_only_root "$PYTHON_TARGET_ROOT"
}

print_config() {
    printf '\nMATLAB SHINE vs SHINIER comparison\n'
    printf '%-24s %s\n' 'repo_root' "$REPO_ROOT"
    printf '%-24s %s\n' 'python_bin' "$PYTHON_BIN"
    printf '%-24s %s\n' 'matlab_bin' "$MATLAB_BIN"
    printf '%-24s %s\n' 'shine_dir' "${SHINE_DIR:-MATLAB path}"
    printf '%-24s %s\n' 'run_root' "$RUN_ROOT"
    printf '%-24s %s\n' 'modes' "$MODES"
    printf '%-24s %s\n' 'limit' "$LIMIT"
    printf '%-24s %s\n' 'iterations' "$ITERATIONS"
    printf '%-24s %s\n' 'full_tracking' "$FULL_TRACKING"
    printf '%-24s %s\n\n' 'seed' "$SEED"
}

PYTHON_BIN="$(prompt_required_path Python "$PYTHON_BIN" PYTHON_BIN)"
MATLAB_BIN="$(detect_matlab_bin "$MATLAB_BIN")"
MATLAB_BIN="$(prompt_required_path MATLAB "$MATLAB_BIN" MATLAB_BIN)"
SHINE_DIR="$(prompt_optional_dir SHINE "$SHINE_DIR" SHINE_DIR)"

COMMON_ARGS=(
    --modes "${MODE_ARGS[@]}"
    --limit "$LIMIT"
    --iterations "$ITERATIONS"
    --seed "$SEED"
    --matlab-bin "$MATLAB_BIN"
    --quiet-run-info
)

if [[ -n "$SHINE_DIR" ]]; then
    COMMON_ARGS+=(--shine-dir "$SHINE_DIR")
fi

if [[ "$FULL_TRACKING" == "1" || "$FULL_TRACKING" == "true" || "$FULL_TRACKING" == "yes" ]]; then
    COMMON_ARGS+=(--full-tracking)
fi

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"
trap cleanup_default_artifacts EXIT
print_config
validate_shine_setup

printf '\n[1/6] Prepare MATLAB vs Python run\n'
"$PYTHON_BIN" "$PY_SCRIPT" "${COMMON_ARGS[@]}" \
    --prepare-only \
    --run-root "$MATLAB_VS_PYTHON_ROOT"

printf '\n[2/6] Run MATLAB for MATLAB vs Python outputs\n'
run_matlab_runner "$MATLAB_VS_PYTHON_ROOT/run_matlab_shine.m"

printf '\n[3/6] Compare MATLAB vs Python outputs\n'
"$PYTHON_BIN" "$PY_SCRIPT" "${COMMON_ARGS[@]}" \
    --skip-matlab \
    --run-root "$MATLAB_VS_PYTHON_ROOT"

printf '\n[4/6] Prepare fixed Python target run\n'
"$PYTHON_BIN" "$PY_SCRIPT" "${COMMON_ARGS[@]}" \
    --use-python-targets \
    --prepare-only \
    --run-root "$PYTHON_TARGET_ROOT"

printf '\n[5/6] Run MATLAB with fixed Python targets\n'
run_matlab_runner "$PYTHON_TARGET_ROOT/run_matlab_shine.m"

printf '\n[6/6] Compare MATLAB/Python against fixed Python targets\n'
"$PYTHON_BIN" "$PY_SCRIPT" "${COMMON_ARGS[@]}" \
    --use-python-targets \
    --skip-matlab \
    --run-root "$PYTHON_TARGET_ROOT"

printf '\nDone. Main outputs:\n'
printf '%-36s %s\n' 'MATLAB vs Python summary' "$MATLAB_VS_PYTHON_ROOT/matlab_shine_comparison_summary.csv"
printf '%-36s %s\n' 'MATLAB vs Python detail' "$MATLAB_VS_PYTHON_ROOT/matlab_shine_comparison_detail.csv"
printf '%-36s %s\n' 'Python target summary' "$PYTHON_TARGET_ROOT/python_target_probe_summary.csv"
printf '%-36s %s\n' 'Python target detail' "$PYTHON_TARGET_ROOT/python_target_probe_detail.csv"
printf '%-36s %s\n' 'Input grayscale comparison' "$MATLAB_VS_PYTHON_ROOT/input_grayscale_comparison.csv"
if [[ "$FULL_TRACKING" == "1" || "$FULL_TRACKING" == "true" || "$FULL_TRACKING" == "yes" ]]; then
    printf '%-36s %s\n' 'Full tracking artifacts' "$RUN_ROOT"
else
    printf '\nIntermediate images/scripts were removed. Set FULL_TRACKING=1 to keep them.\n'
fi
