# NeuroDose-NJ: Clinical Policy & Deep Learning Integration

The NeuroDose-NJ project represents a sophisticated intersection of clinical policy and high-end deep learning.
Implemented in MATLAB, it uses the Deep Learning, Reinforcement Learning, and Statistics and Machine Learning toolboxes
to provide a unified environment for multi-modal data processing.

## Project Architecture: The LIGHTED Framework
This hybrid architecture processes both relational (provider-patient networks) and temporal (dose history) data.

### Key DNN Components
1. **Graph Neural Network (GNN)** (`src/models/build_gnn_model.m`): Models the "Care Coordination" aspect by treating patients, doctors, and pharmacies as nodes to detect "doctor shopping" or high-risk prescription clusters. It extracts a global node representation for fusion.
2. **Long Short-Term Memory (LSTM)** (`src/models/build_lstm_model.m`): Captures the temporal trajectories of patient health. The architecture utilizes a deep sequence network consisting of 3 hidden LSTM layers (sizes 128, 64, and 32) interspersed with Dropout layers (0.2) for regularization, culminating in a classification layer to predict overdose risk.
3. **Actor-Critic Reinforcement Learning (RL)** (`src/rl_agent/setup_rl_agent.m`): The "Decision Support" layer that suggests optimal dose titrations while maintaining "Safe RL" constraints to prevent respiratory depression. It utilizes two distinct, fully connected neural networks:
   - **Critic Network**: Evaluates the state value `V(s)`.
   - **Actor Network**: Defines the policy `pi(a|s)` for proposing discrete dose changes.

## Project Structure
- `data/`: Contains datasets, including the CDC drug overdose death rates.
- `src/preprocessing/`: Contains scripts for data integration and feature engineering (`preprocess_data.m`).
- `src/models/`: Contains model building scripts for the LSTM (`build_lstm_model.m`) and GNN (`build_gnn_model.m`).
- `src/rl_agent/`: Contains the Reinforcement Learning agent setup (`setup_rl_agent.m`).
- `src/main.m`: The main pipeline script that ties preprocessing, modeling, and RL together.

## Running the Code
1. Ensure MATLAB is installed with the following toolboxes:
   - Deep Learning Toolbox
   - Reinforcement Learning Toolbox
   - Statistics and Machine Learning Toolbox
   - Text Analytics Toolbox (for network analysis)
2. Open the project directory in MATLAB.
3. Run `src/main.m` to execute the full pipeline using the simulated dataset and stub models.
