import argparse
import os
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Optional
from plotUtils2 import MRIDataProcessor, MRIPlotter
from plotConfig import PlotConfig

def load_config(config_path: str) -> PlotConfig:
    """Loads the configuration from a JSON or YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config_data = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format. Use JSON or YAML.")
    return PlotConfig(**config_data)

def save_config(config_path: str, config: PlotConfig):
    """Saves the configuration to a JSON or YAML file."""
    config_data = asdict(config)
    with open(config_path, 'w') as f:
        if config_path.endswith('.json'):
            json.dump(config_data, f, indent=4)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.safe_dump(config_data, f)
        else:
            raise ValueError("Unsupported config file format. Use JSON or YAML.")

def extract_scan_name(mri_data_path: str) -> str:
    """Extracts the scan name from the file name in the MRI data path."""
    return os.path.splitext(os.path.basename(mri_data_path))[0]

def main():
    parser = argparse.ArgumentParser(description="MRI Plotter CLI Tool")
    parser.add_argument("-i", "--mri_data_path", required=True, help="Path to the MRI data file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save the output.")
    parser.add_argument("-c", "--config_path", help="Path to the configuration file (JSON or YAML).")
    parser.add_argument("-u", "--underlay_image_path", help="Path to the underlay image (optional).")
    parser.add_argument("-k", "--mask_path", help="Path to the mask image (optional).")
    parser.add_argument("--edit_config", action="store_true", help="Open the configuration file for editing.")

    args = parser.parse_args()

    # Extract the scan name from the MRI data path
    scan_name = extract_scan_name(args.mri_data_path)

    # Edit configuration if requested
    if args.edit_config:
        default_config = PlotConfig()
        default_config_path = os.path.join(os.getcwd(), "default_config.yaml")
        save_config(default_config_path, default_config)
        print(f"Default configuration written to {default_config_path}.")
        return

    # Load configuration or use defaults
    if args.config_path:
        config = load_config(args.config_path)
    else:
        print("No configuration file provided. Using default configuration.")
        config = PlotConfig()

    # Initialize and preprocess the MRI data
    processor = MRIDataProcessor(
        mri_data_path=args.mri_data_path,
        config=config,
        underlay_image_path=args.underlay_image_path,
        mask_path=args.mask_path
    )

    # Initialize and run the plotter
    plotter = MRIPlotter(
        media_type=processor.media_type,
        mri_data=processor.mri_slices,
        config=config,
        output_dir=args.output_dir,
        scan_name=scan_name,
        underlay_image=processor.underlay_slices
    )
    plotter.plot()
    print(f"Plotting complete. Outputs saved to {args.output_dir}.")

if __name__ == "__main__":
    main()