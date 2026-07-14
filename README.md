# Evolutionary Lunar Lander

Academic project developed for the **Fundamentals of Artificial Intelligence** course at the **University of Coimbra**.

## Overview

This project explores the use of **neuroevolution** to control an autonomous Lunar Lander agent in the **Gymnasium LunarLander-v3** environment.

A feed-forward neural network is evolved using a genetic algorithm to maximise the agent's landing performance. Different evolutionary configurations were evaluated and compared through multiple experimental runs.

---

## Features

- Feed-forward neural network controller
- Evolutionary optimisation of network weights
- Tournament selection
- Two-point crossover
- Gaussian mutation
- Elitist survival strategy
- Custom fitness function
- Performance evaluation over multiple generations

---

## Technologies

- Python
- NumPy
- Gymnasium
- Matplotlib

---

## Project Structure

```text
.
├── src/
│   ├── reactive_agent.py
│   └── evolutionary_agent.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running

Run the reactive controller:

```bash
python src/reactive_agent.py
```

Run the evolutionary controller:

```bash
python src/evolutionary_agent.py
```

---

## Results

The project compares different evolutionary configurations by analysing:

- Best fitness evolution
- Average population fitness
- Successful landing rate
- Agent performance across multiple independent runs

Experimental plots and analysis are available in the project report.

---

## Authors

- Diogo André de Freitas Alves
- Bruno Miguel Santos Marques

---

## Acknowledgements

Developed as an academic group project for the **Fundamentals of Artificial Intelligence** course at the **University of Coimbra**.
