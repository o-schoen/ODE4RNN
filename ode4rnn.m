function [T, D] = ode4rnn(predictFcn, controlFcn, tSpan, stepSize4Rnn, x_0, maxSequenceLength, UsePlotting)
if nargin < 7
    UsePlotting = false;
end

% Time steps for the RNN
T4ODE = tSpan(1):stepSize4Rnn:tSpan(2);

% Init closure object
[odeFcn, outputFcn] = closure(predictFcn, controlFcn, maxSequenceLength, tSpan, stepSize4Rnn, UsePlotting);

% Pass custom output function through odeset
opts = odeset('OutputFcn', outputFcn);

% Solve ODE
[T, X] = ode45(odeFcn, T4ODE, x_0, opts);

% Return results
U = controlFcn(T');
D = [X'; U];
end

% Function closure for broadcasting driver and time histories to several
% functions in use
function [odeFcn, outputFcn] = closure(predictFcn, controlFcn, maxSequenceLength, tSpan, stepSize4Rnn, UsePlotting)
% Init driver and time history
driverHist = dlarray([], 'CT');
timeHist = [];

% Return function handles
odeFcn = @odeFcn_;
outputFcn = @outputFcn_;

%% Functions
    % Custom output function for fetching accepted intermediary data from
    % the ode45 solver
    function status = outputFcn_(T, X, flag)
        status = [];
        
        if isempty(flag) % Valid data incoming
            % Calculate control for new time steps
            U = controlFcn(T);
            
            % Add new data to history
            driverHist = [driverHist, [X; U]];
            timeHist = [timeHist, T];

            % Check length of history
            [driverHist, timeHist] = historyLengthChecker(driverHist, timeHist);
        end
    end

    % Custom ode function calculating the response of the RNN
    function dxdt_k = odeFcn_(t_k, x_k)
        % Calculate network response (predict method normalizes input data)
        % But first check if driver history contains any data
        if isempty(driverHist)
            % History is empty -> generate auxiliary data
            u_k = controlFcn(t_k);
            d_k = dlarray([x_k; u_k], 'CT');
            
            % Calculate response
            dxdt_k = predictNetwork(d_k);
        else
            % History contains data -> interpolate history
            interpDriverHist = interpolateHistory(t_k, x_k);
            
            dxdt_k = predictNetwork(interpDriverHist);
        end
    end

    % Function takes sequence of network drivers (dlarray) and calculates network
    % response while giving back the response for the last time step (double)
    function y_k = predictNetwork(X)
        Y = double(gather(extractdata(predictFcn(X))));
%         Y = double(gather(extractdata(rnn.predict(X, nn))));
        y_k = Y(:, end);
    end

    % Due to ode45 not adhering to the stepSize4Rnn time grid, in order of
    % supplying network with correctly spaced history, history has to be
    % anchored by newest time stamp and interpolated to meet the grid
    function interpHistory = interpolateHistory(t_k, x_k)
        % Check if interpolation is necessary
        if t_k == timeHist(end) + stepSize4Rnn
            % No interpolation necessary because matches stepSize4Rnn
            interpHistory = driverHist;
        else
            % Interpolation necessary because doesn't match stepSize4Rnn
            % Create time grid to meet stepSize4Rnn
            timeGrid = flip(t_k:-stepSize4Rnn:tSpan(1));
            
            % Check length of history
            timeGrid = timeHistoryLengthChecker(timeGrid);
            
            % Build current driver
            u_k = controlFcn(t_k);
            d_k = [x_k; u_k];
            
            % Interpolate
            T_temp = [timeHist, t_k];
            D_temp = [double(gather(extractdata(driverHist))), d_k];
            temp = interp1(T_temp', D_temp', timeGrid', 'pchip', 'extrap')';
            
            % Plot data
            if UsePlotting
                plot(T_temp', D_temp', 'k-o', 'MarkerSize', 5, 'MarkerFaceColor', 'k', 'LineWidth', 1.5)
                hold on
                xlim(tSpan)
                ylim([-5, 8])
                grid on
                plot(timeGrid', temp', 'c--o', 'MarkerSize', 10, 'LineWidth', 1.5)
                hold off
                drawnow limitrate
            end
            
            interpHistory = dlarray( ...
                temp, ...
                'CT');          
        end
    end

    % Function checks time history to stay below the maximum sequence
    % length allowed
    function historyT = timeHistoryLengthChecker(historyT)
        if length(historyT) > maxSequenceLength
            % Pop oldest entries
            historyT = historyT(end-maxSequenceLength+1:end);
        end
    end

    % Function checks time and driver histories to stay below the maximum 
    % sequence length allowed
    function [historyD, historyT] = historyLengthChecker(historyD, historyT)
        if length(historyT) > maxSequenceLength
            % Pop oldest entries
            historyD = historyD(:, end-maxSequenceLength+1:end);
            historyT = historyT(end-maxSequenceLength+1:end);
        end
    end
end