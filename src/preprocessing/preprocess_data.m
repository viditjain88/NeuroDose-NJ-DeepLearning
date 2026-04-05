function [cleanData, featureData] = preprocess_data(dataFilePath)
% PREPROCESS_DATA Phase I: Data Integration & Feature Engineering
%
% This function simulates the integration of EHR, PMP data, and SDoH
% as required for the NeuroDose-NJ project. It uses MATLAB's datastore
% and tall arrays for handling large-scale datasets (like the CDC overdose data).
%
% Inputs:
%   dataFilePath: Path to the raw dataset (e.g., CSV file).
%
% Outputs:
%   cleanData: Tall table containing preprocessed and cleaned data.
%   featureData: Array or table of normalized features ready for modeling.

    disp('Starting Phase I: Data Integration & Preprocessing...');

    if nargin < 1
        % Default to a placeholder if no path is provided (assuming run from root)
        dataFilePath = 'data/overdose_data.csv';
    end

    % 1. Create a datastore for large-scale data
    try
        ds = datastore(dataFilePath, 'TreatAsMissing', 'NA');
        disp(['Successfully created datastore from: ', dataFilePath]);
    catch ME
        warning(['Could not load data from ', dataFilePath, '. Creating simulated data datastore...']);
        % Create mock data for simulation purposes if the file doesn't exist
        mockData = array2table(randn(1000, 10), 'VariableNames', ...
            {'Age', 'PainLevel', 'MME_Current', 'PastOverdoses', ...
             'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'TargetRisk'});

        % Introduce some missing values for fillmissing to handle
        mockData.PainLevel(randi(1000, 50, 1)) = NaN;
        mockData.MME_Current(randi(1000, 20, 1)) = NaN;

        writetable(mockData, 'data/simulated_data.csv');
        ds = datastore('data/simulated_data.csv');
    end

    % 2. Convert datastore to a tall array for out-of-memory computation
    tt = tall(ds);

    % 3. Preprocessing: Handle missing values
    % Using 'fillmissing' with 'constant' or 'previous' methods
    % Note: In tall arrays, some fillmissing methods are restricted.
    % We use constant imputation or simple mean if supported, here demonstrating basic removal/fill.

    % For demonstration, we simulate filling missing numerical values with 0
    % In a real EHR scenario, this might be clinical baselines.
    cleanTallData = fillmissing(tt, 'constant', 0);

    % 4. Normalize physiological data
    % Since it's a tall array, we calculate statistics first
    % Here we assume columns 1 to 4 are our main features for normalization in our mock

    disp('Normalizing features...');
    % Evaluate the tall array to bring results into memory (simulated small batch here)
    cleanData = gather(cleanTallData);

    % Basic Z-score normalization for numerical columns
    numCols = vartype('numeric');
    numericData = cleanData(:, numCols);
    featureData = normalize(numericData{:,:}, 'zscore');

    disp('Phase I Preprocessing complete. Data ready for Phase II.');
end
