function detected_events = swr_eventDetection(data, labels, baseline_window, past_window, verification_window, grace_period, dynamic_threshold, max_trend_threshold, slope_threshold)
    % Parameters
    baseline_avg = movmean(data, [baseline_window, 0]); % Moving average baseline (past window only)
    dynamic_threshold = 0.1; % Starting threshold (modifiable)
    detected_events = []; % Store detected events
    positive_trend = 0; % Trend check counter
    max_trend_threshold = 3; % Trend required over the window to confirm
    confidence_score = 0; % Confidence score
    slope_threshold = 0.02; % Minimum slope to consider a positive trend
    
    % Loop over data for real-time detection
    i = 1;
    while i <= length(data) - verification_window
        % Calculate past baseline at the current point
        past_baseline = baseline_avg(max(i - past_window, 1));

        % Step 1: Check for an event onset
        if data(i) > past_baseline + dynamic_threshold
            % Confidence mechanism - track positive trend in the verification window
            pos_trend_window = data(i:i + verification_window - 1); % Lookahead for trend checking
            slope = mean(diff(pos_trend_window)); % Calculate trend slope
            avg_prob = mean(pos_trend_window); % Avg probability across the verification window
            
            if avg_prob > dynamic_threshold && slope > slope_threshold
                % Step 2: Verification phase (positive trend detected)
                confidence_score = confidence_score + 1; % Increment confidence
                positive_trend = positive_trend + 1; % Increment trend
                
                % Detection trigger based on rapid confidence score or trend
                if (confidence_score >= max_trend_threshold) || avg_prob > 0.75
                    fprintf('Event detected at index %d\n', i);
                    detected_events = [detected_events, i];
                    
                    % Update parameters
                    if any(labels(i:i + grace_period) == 1) % Check for label match within grace
                        dynamic_threshold = max(dynamic_threshold - 0.05, 0.1); % Lower if missed event
                    else
                        dynamic_threshold = min(dynamic_threshold + 0.05, 0.9); % Raise threshold if false positive
                    end
                    
                    % Apply grace period after detection
                    i = i + grace_period; % Skip ahead to avoid re-detection
                    confidence_score = 0; % Reset confidence
                    positive_trend = 0; % Reset trend counter
                end
            else
                % No positive trend or confidence fail - reset trend and threshold
                confidence_score = max(confidence_score - 1, 0); % Lower confidence
                dynamic_threshold = min(dynamic_threshold * 1.02, 0.9); % Gradual threshold raise
            end
        end
        i = i + 1;
    end
    
    % Plot results
% %     figure;
% %     plot(data, 'b'); hold on;
% %     plot(baseline_avg, 'g--');
% %     for event = detected_events
% %         xline(event, 'r', 'LineWidth', 1.5);
% %     end
%     title('Event Detection with Dynamic Threshold and Trend Verification');
%     xlabel('Time');
%     ylabel('Signal Probability');
%     legend('Data', 'Baseline', 'Detected Events');
end

% Function to optimize parameters for event detection
function [optimal_params, performance] = optimize_eventDetection(data, labels, baseline_window, past_window, verification_window, grace_period)
    % Define parameter ranges
    threshold_range = 0.05:0.05:0.2;
    max_trend_threshold_range = 1:5;
    slope_threshold_range = 0.01:0.01:0.1;
    
    best_performance = -inf;
    optimal_params = struct('threshold', 0, 'max_trend_threshold', 0, 'slope_threshold', 0);
    
    % Grid search over parameter ranges
    for threshold = threshold_range
        for max_trend_threshold = max_trend_threshold_range
            for slope_threshold = slope_threshold_range
                % Run event detection with current parameters
                detected_events = swr_eventDetection(data, labels, baseline_window, past_window, verification_window, grace_period, threshold, max_trend_threshold, slope_threshold);
                
                % Evaluate performance
                [TP, FP] = evaluate_performance(detected_events, labels, grace_period);
                performance = TP - FP; % Simple metric: maximize TP, minimize FP
                
                % Update best parameters if performance improves
                if performance > best_performance
                    best_performance = performance;
                    optimal_params.threshold = threshold;
                    optimal_params.max_trend_threshold = max_trend_threshold;
                    optimal_params.slope_threshold = slope_threshold;
                end
            end
        end
    end
end

% Function to evaluate performance
function [TP, FP] = evaluate_performance(detected_events, labels, grace_period)
    TP = 0;
    FP = 0;
    for event = detected_events
        if any(labels(event:event + grace_period) == 1)
            TP = TP + 1;
        else
            FP = FP + 1;
        end
    end
end