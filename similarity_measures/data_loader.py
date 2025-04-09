import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yaml_data(filepath: str) -> list | None:
    """Loads and parses the YAML data file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        logging.info(f"Successfully loaded data from {filepath}")
        if not isinstance(data, dict) or 'questions' not in data:
             logging.error(f"YAML file {filepath} must have a root 'questions' key containing a list.")
             return None
        return data.get('questions', []) # Return the list of questions
    except FileNotFoundError:
        logging.error(f"Error: YAML file not found at {filepath}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {filepath}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {filepath}: {e}")
        return None

def extract_thoughts_from_section(section: dict) -> list[str]:
    """Extracts all 'thought' strings from a section's responses."""
    thoughts = []
    if 'responses' in section and isinstance(section['responses'], list):
        for response in section['responses']:
            if isinstance(response, dict) and 'thought' in response and isinstance(response['thought'], str):
                thoughts.append(response['thought'].strip())
    return thoughts