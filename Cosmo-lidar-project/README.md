# Cosmo-lidar Project

## Overview
The Cosmo-lidar project focuses on atmospheric calculations and functions, utilizing lidar technology to analyze and interpret atmospheric data. This project includes various tools and utilities for processing and analyzing atmospheric data.

## Project Structure
```
Cosmo-lidar-project
├── notebooks
│   └── code-propre-valer.ipynb
├── src
│   └── cosmo_lidar
│       ├── __init__.py
│       ├── atm_tools.py
│       └── utils.py
├── data
│   ├── raw
│   └── processed
├── tests
│   └── test_atm_tools.py
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation
To set up the project environment, you can use either `conda` or `pip`. 

### Using Conda
1. Create a new conda environment:
   ```
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```
   conda activate cosmo-lidar
   ```

### Using Pip
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
- The main functionalities of the project can be accessed through the `src/cosmo_lidar` package.
- The Jupyter notebook located in `notebooks/code-propre-valer.ipynb` provides examples and documentation on how to use the functions and tools available in the project.

## Testing
Unit tests for the project can be found in the `tests/test_atm_tools.py` file. To run the tests, use:
```
pytest tests/test_atm_tools.py
```

## Contributing
Contributions to the Cosmo-lidar project are welcome. Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.