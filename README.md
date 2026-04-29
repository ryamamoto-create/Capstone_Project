## Capstone Project Structure
### Project contents
* README.md
* pyproject.toml
  * Setup for this project
  * Run [pip install -e .] to install packages
* .vscode/
  * settings.json
    * Settings for VSCode
* .gitignore
  * Ignore data and configurations
* data/
  * raw/
    * data.txt
      * Contains movie rating information
    * movieTitles.csv
      * Contains movie titles and IDs
  * processed/
    * ratings.parquet
      * A parquet containing the parsed data.txt file
* notebooks/
  * exploration.ipynb
    * Used for exploring data before implementing
* src/
  * \_\_init__.py
    * Treat this directory as a package
  * config.py
    * N-number and paths to data
  * parsing.py
    * Getting the data into the dataframes and parquet
  * preprocessing.py
    * Getting the data into a more workable state
  * baseline.py
    * Train baseline models
  * model.py
    * Train improved model(s)
  * evaluation.py
    * Evaluate the models
* img/
  * Images
* main.py
  * Main python file