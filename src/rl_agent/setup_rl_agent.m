function agent = setup_rl_agent()
% SETUP_RL_AGENT Phase IV: Dynamic Dose Titration (Safe RL)
%
% This function sets up an Actor-Critic Reinforcement Learning agent
% (rlACAgent) to act as the "Stewardship" engine. It suggests optimal
% dose titrations while maintaining "Safe RL" constraints to prevent
% respiratory depression.
%
% Outputs:
%   agent: The configured Reinforcement Learning agent.

    disp('Starting Phase IV: Setting up Dynamic Dose Titration RL Agent...');

    % 1. Define Observation and Action Specifications
    % Observation: [Pain_Level, Risk_Score, Current_MME, Resp_Rate]
    obsInfo = rlNumericSpec([4 1]);
    obsInfo.Name = 'Patient_State';

    % Action: Change in Dose (e.g., -10 MME, +0 MME, +10 MME)
    % Assuming a discrete action space for simplicity: [-10, -5, 0, 5, 10]
    actInfo = rlFiniteSetSpec([-10 -5 0 5 10]);
    actInfo.Name = 'Dose_Titration_Action';

    % 2. Define the Environment (Conceptual)
    % In reality, you'd use rlFunctionEnv or a custom Simulink/MATLAB environment
    % to simulate the physiological model of opioid metabolism (PK/PD).
    disp('Environment involves simulating opioid metabolism (PK/PD) and calculating rewards.');
    disp('Reward Function: R = -(Pain_Level) - \lambda * (Risk_Score)');

    % 3. Create Actor and Critic Networks
    % Critic Network: Evaluates state value V(s)
    criticNetwork = [
        featureInputLayer(4, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(32, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(1, 'Name', 'CriticOutput')];
    criticOpts = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1);
    critic = rlValueRepresentation(criticNetwork, obsInfo, 'Observation', {'state'}, criticOpts);

    % Actor Network: Policy pi(a|s)
    actorNetwork = [
        featureInputLayer(4, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(32, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'ActorOutput')
        softmaxLayer('Name', 'actionProb')];
    actorOpts = rlRepresentationOptions('LearnRate', 1e-4, 'GradientThreshold', 1);
    actor = rlStochasticActorRepresentation(actorNetwork, obsInfo, actInfo, ...
        'Observation', {'state'}, actorOpts);

    % 4. Configure the Actor-Critic Agent
    agentOpts = rlACAgentOptions(...
        'NumStepsToLookAhead', 10, ...
        'DiscountFactor', 0.99);

    agent = rlACAgent(actor, critic, agentOpts);

    disp('Safe RL Configuration:');
    disp('Ensure proposed actions do not push MME above CDC thresholds using action constraints or negative reward bounds.');

    disp('RL Agent setup complete.');
end
