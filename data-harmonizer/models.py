from pydantic import BaseModel
from typing import List
import yaml


class FieldMapping(BaseModel):
    """Represents a single field mapping between source and target."""
    source_field: str
    target_field: str


class DataMapping(BaseModel):
    """Represents a complete data mapping configuration."""
    origin: str
    dest: str
    fields_map: List[FieldMapping]


class ConfigRoot(BaseModel):
    """Represents the root configuration with target classes and datasets."""
    target_classes: List[str]
    datasets: List[DataMapping]


def load_config_mappings(config_path: str) -> ConfigRoot:
    """
    Load configuration from YAML file and convert to ConfigRoot Pydantic object.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ConfigRoot Pydantic object with target_classes and datasets
    """
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # Convert datasets to Pydantic objects
    datasets = []
    for mapping_config in config_data["datasets"]:
        # Convert each mapping to Pydantic objects
        field_mappings = [
            FieldMapping(
                source_field=field["source_field"],
                target_field=field["target_field"]
            )
            for field in mapping_config["fields_map"]
        ]
        
        data_mapping = DataMapping(
            origin=mapping_config["origin"],
            dest=mapping_config["dest"],
            fields_map=field_mappings
        )
        datasets.append(data_mapping)
    
    # Create the root configuration object
    config_root = ConfigRoot(
        target_classes=config_data["target_classes"],
        datasets=datasets
    )
    
    return config_root


# Example usage
if __name__ == "__main__":
    # Load the configuration
    config = load_config_mappings("config.yml")
    
    # Print the loaded configuration
    print(f"Target Classes: {config.target_classes}")
    print(f"Number of datasets: {len(config.datasets)}")
    print()
    
    # Print the loaded mappings
    for i, mapping in enumerate(config.datasets):
        print(f"Dataset {i+1}:")
        print(f"  Origin: {mapping.origin}")
        print(f"  Destination: {mapping.dest}")
        print(f"  Field mappings:")
        for field in mapping.fields_map:
            print(f"    {field.source_field} -> {field.target_field}")
        print()
