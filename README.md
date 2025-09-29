# 8-Puzzle Solver

This project is an interactive 8-puzzle game built with Python and Streamlit.  
Users can upload an image, slice it into 9 tiles, shuffle the puzzle into a solvable configuration, and either play manually or let the program solve it automatically.

The solver uses the A* search algorithm with a Manhattan distance heuristic to guarantee an optimal solution.  
For comparison, BFS and DFS implementations are also included to show the tradeoffs between different search strategies.

## Features
- Upload an image and automatically slice it into tiles  
- Shuffle with customizable difficulty  
- Manual play with arrow buttons in the sidebar  
- Step-by-step autoplay of the solution  
- Download the solution path as JSON  
- Multiple solver options: A*, BFS, DFS  

## Installation and Usage

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/8-puzzle-solver.git
cd 8-puzzle-solver
python -m pip install streamlit pillow
```
## Run the app locally
```bash
python -m streamlit run app.py
```
## Website Access

https://cai-4002-assignment-1-b27yaza9aeduhrerluaehu.streamlit.app/
