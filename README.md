## Capstone Project Structure
### Project contents
* README.md
* requirements.txt
  * Required packages for this project
* .gitignore
  * Ignore data and configurations
* config.py
  * N-number and path to movie rating data (too large to upload)
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
  * parsing.py
    * Getting the data into the dataframes and parquet
  * preprocessing.py
    * Getting the data into a more workable state
  * split.py
    * Train-test split
  * baseline.py
    * Train baseline model
  * model.py
    * Train improved model(s)
  * evaluation.py
    * Evaluate the models
  * utils.py
    * Utility functions
* main.py
  * Main python file