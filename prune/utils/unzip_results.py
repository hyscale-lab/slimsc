#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path
import zipfile

def unzip_results(model: str, dataset: str, dir_name: str, verbose: bool):
    """
    Unzips a specified result directory without overwriting existing files.
    Allows controlling verbosity of detailed file extraction/skipping messages.

    Args:
        model (str): The model name (e.g., QwQ-32B).
        dataset (str): The dataset name (e.g., hmmt).
        dir_name (str): The name of the directory/zip file (e.g., sc_8_control).
        verbose (bool): If True, prints detailed messages for each extracted/skipped file.
    """

    # --- Configuration ---
    script_dir = Path(__file__).resolve().parent
    base_results_dir = (script_dir / "../results").resolve()

    # --- Construct Paths ---
    target_parent_dir = base_results_dir / model / dataset
    zip_file_path = target_parent_dir / f"{dir_name}.zip"
    target_extraction_root = target_parent_dir # Extract into the parent directory

    print(f"--- Unzipping Results ---")
    print(f"Model: {model}, Dataset: {dataset}, Directory: {dir_name}")
    if verbose:
        print(f"Base Results Dir: {base_results_dir}")
        print(f"Target Parent Dir: {target_parent_dir}")
        print(f"ZIP File Path: {zip_file_path}")
        print(f"Extraction Root: {target_extraction_root}")

    # --- Validation ---
    if not target_parent_dir.is_dir():
        print(f"Error: Parent directory not found: {target_parent_dir}", file=sys.stderr)
        print("Please ensure the model and dataset paths are correct and exist.", file=sys.stderr)
        sys.exit(1)

    if not zip_file_path.is_file():
        print(f"Error: ZIP file not found: {zip_file_path}", file=sys.stderr)
        print("Please ensure the .zip file exists at the specified path.", file=sys.stderr)
        sys.exit(1)

    # --- Unzip Operation ---
    print(f"\nAttempting to unzip: {zip_file_path}")
    print(f"Into: {target_extraction_root}")
    print("Unzipping without overwriting existing files...")

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            members = zip_ref.namelist()

            if not members:
                print(f"Warning: ZIP file '{zip_file_path}' is empty.", file=sys.stderr)
                return

            extracted_count = 0
            skipped_count = 0

            for member in members:
                target_member_path = target_extraction_root / member

                # Determine if it's a file or a directory (zip members ending with '/' are directories)
                is_file = not member.endswith('/')
                
                # Check for existing files to skip
                if is_file and target_member_path.exists():
                    if verbose:
                        print(f"  - Skipping existing file: {member}")
                    skipped_count += 1
                    continue # Skip to the next member in the loop

                try:
                    # Extract the member. For directories, this just ensures the path exists.
                    # For files, it extracts them, but we've already handled skipping existing files.
                    zip_ref.extract(member, target_extraction_root)
                    
                    if is_file: # Only count and print for actual files, not directories
                        if verbose:
                            print(f"  + Extracted: {member}")
                        extracted_count += 1
                    # Directories are simply ensured to exist; no verbose output for them unless specifically desired.
                    
                except Exception as e:
                    print(f"  ! Error extracting '{member}': {e}", file=sys.stderr)
                    # Don't exit, try to continue with other files
            
            print(f"\nUnzip complete for {dir_name}.zip")
            print(f"Summary: {extracted_count} files extracted, {skipped_count} files skipped (already existed).")

    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid ZIP file or is corrupted.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: ZIP file not found or inaccessible: {zip_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during unzipping: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unzips result directories without overwriting existing files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model",
        help="The model name (e.g., QwQ-32B)"
    )
    parser.add_argument(
        "dataset",
        help="The dataset name (e.g., hmmt)"
    )
    parser.add_argument(
        "directory_name",
        help="The name of the directory/zip file (e.g., sc_8_control or random_n8_thresh0.92_delay20)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", # This makes it a boolean flag (default False if not present)
        help="Enable verbose output (show details for each extracted/skipped file)."
    )

    args = parser.parse_args()

    # Pass the 'verbose' argument to the function
    unzip_results(args.model, args.dataset, args.directory_name, args.verbose)