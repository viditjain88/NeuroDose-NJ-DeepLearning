function lgraph = build_lstm_model(numFeatures, numClasses)
% BUILD_LSTM_MODEL Phase II: Building the Sequential Model (LSTM)
%
% This function defines a deep LSTM network architecture intended to capture
% the temporal trajectories of patient health (e.g., pain levels, withdrawal
% symptoms, and dose escalation over 12 months) to predict overdose risk.
%
% Inputs:
%   numFeatures: Number of input features from the preprocessed temporal data.
%   numClasses: Number of output classes (e.g., risk categories).
%
% Outputs:
%   lgraph: Layer graph defining the LSTM network architecture.

    disp('Starting Phase II: Building Sequential LSTM Model...');

    if nargin < 2
        numFeatures = 10; % Default simulated features
        numClasses = 2;   % Default binary risk: Low Risk vs High Risk
    end

    % Define the Deep LSTM architecture with 3 hidden layers and Dropout
    layers = [ ...
        sequenceInputLayer(numFeatures, 'Name', 'seq_input')

        lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'lstm_1')
        dropoutLayer(0.2, 'Name', 'dropout_1')

        lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'lstm_2')
        dropoutLayer(0.2, 'Name', 'dropout_2')

        lstmLayer(32, 'OutputMode', 'last', 'Name', 'lstm_3')

        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classification')
    ];

    lgraph = layerGraph(layers);

    disp('LSTM architecture defined.');
    disp('Target Objective: Achieve C-statistic of 0.91 for overdose risk identification.');

    % NOTE: In the main training script, use 'ExecutionEnvironment', 'gpu'
    % with trainNetwork() to utilize hardware acceleration.
end
