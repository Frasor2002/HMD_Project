<div align="center">

# Human-Machine Dialogue Project

</div>

Project for the "Human-Machine Dialogue Project" course at University of Trento.

## Installation and Execution
Follow the following steps to run the project code:

1. Clone the repository:
   ```sh
   git clone https://github.com/Frasor2002/BIOIAI_Project.git
   cd HMD_Project
   ```

2. Install modules:
   ```sh
   pip install -r requirements.txt
   ```

3. Set the environment variable HF_TOKEN with the personal token.

4. To run the project the command is:
    ```sh
    python main.py
   ```

## Evaluation
The evaluation is executed with the following command.
```sh
   python evaluate.py
```

The parameters are:
- `model`: model to test, set with --model or -m .
- `component`: component to test, set with --component or -c.

## Dataset
This project uses the Steam Games 2025 Dataset on Kaggle, this repository only has a trimmed down version as an example for storage constraints. 

The complete version can be dowloaded from https://www.kaggle.com/datasets/artermiloff/steam-games-dataset/data and then needs to be converted into feather format for faster computation.

The project also uses the videogame glossary page to get knowledge on videogame terminology. This can be found at https://en.wikipedia.org/wiki/Glossary_of_video_game_terms.