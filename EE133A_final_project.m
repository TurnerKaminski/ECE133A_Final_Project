clear; clc; close all;

% This code is now partitioned into sections. To see results from part 
% 3a, 3b, 3c, or 3d, simply run this initial section to setup the data
% matrix and then run the corresponding part. This only needs to be done
% once.

%%Part 1

% element_data is a breakdown of each superconductor into its chemical
% formula by element, also listing its critical temp and the written 
% chemical formula
element_data = readtable("unique_m.csv");
% superconductor_data gives the feature values of the same superconductors
% that are in element_data
superconductor_data = readtable("train.csv");

% For running Part 3
X_matrix = superconductor_data{:,:};
X_standardized = normalize(X_matrix);

%{
% The following section is just us looking at some basic plots and info 
% from our tables to help understand the data, commented out

% Histogram of the critical temps from first table
histogram(element_data.("critical_temp")(1:21263))
figure()
% Histogram of critical temps from second table to confirm they're the same
histogram(superconductor_data.("critical_temp")(1:21263))
figure()

% Plot the percentage of superconductors with each element
x = 1:86;
elements = element_data(:,1:86);
for i = 1:86
    table(i) = (sum(elements.(i) ~= 0))/21263;
end

% Store the table as an array so we can associate each value with its
% element
table = array2table(table);
table.Properties.VariableNames = elements.Properties.VariableNames;
[~, idx] = sort(table{:,:}, 'descend');
table = table(:,idx);
% Plotting stuff
scatter(x,table{:,:});
text(x,table{:,:},table.Properties.VariableNames)
xlim([0 86])

% Find the standard deviation and mean of critical temp for each element
% whereever there is a non-zero entry, add the corresponding crit temp
for i = 1:86
    temp = 0;
    non_zero_entry_indices = find(elements.(i));
    if(~isempty(non_zero_entry_indices))
        for t = 1:length(non_zero_entry_indices)
            index_value = non_zero_entry_indices(t);
            temp(t) = element_data.("critical_temp")(index_value);
        end
    end
    means(i) = mean(temp);
    stds(i) = std(temp);
end

% Put the arrays into tables and sort them in descending order
means = array2table(means);
stds = array2table(stds);
means.Properties.VariableNames = elements.Properties.VariableNames;
stds.Properties.VariableNames = elements.Properties.VariableNames;
[~, idx] = sort(means{:,:}, 'descend');
means = means(:,idx);
[~, idx] = sort(stds{:,:}, 'descend');
stds = stds(:,idx);
% Plot the mean and std
figure()
scatter(x,means{:,:});
text(x, means{:,:}, means.Properties.VariableNames)
figure()
scatter(x,stds{:,:});
text(x, stds{:,:}, stds.Properties.VariableNames)

% Plot the mean vs std both linear and log
figure()
scatter(means{:,:}, stds{:,:});
figure()
scatter(log(means{:,:}), stds{:,:})
%}

%% Part 2

% Remove these bracket comments to run part 2
%{

% Create an unlabeled matrix to perform standardization
% Find mean and std of each feature
X_matrix = superconductor_data{:,:};
X_standardized = normalize(X_matrix);
mean_features = varfun(@mean, superconductor_data, 'InputVariables', @isnumeric);
std_features = varfun(@std, superconductor_data, 'InputVariables', @isnumeric);
% Perform k-means clustering
for t = 1:10
k = t;
% Perform k-means clustering on the features
[idx, centroids] = kmeans(X_standardized(:,1:81), k);
crit_temps = X_matrix(:, end);
% Compute the predicted critical temperatures
pred_crit_temps = zeros(size(crit_temps));
for i = 1:k
    cluster_samples = find(idx == i);
    pred_crit_temps(cluster_samples) = mean(crit_temps(cluster_samples));
end
% Calculate the RMSE between the actual and predicted critical temperatures
rmse = sqrt(mean((crit_temps - pred_crit_temps).^2));
end
%perform SVD, S is a 82x1 array of the SVD values in descending order
S = svd(X_standardized);
%find the correlation matrix of the standardized data
%Create a table to see which features best correlate to critical temp
cor_matrix = corr(X_standardized);
cor_to_crit_temp = cor_matrix(:,82);
best_cor = abs(cor_to_crit_temp);
best_cor = array2table(transpose(best_cor));
best_cor.Properties.VariableNames = superconductor_data.Properties.VariableNames;
[~, idx] = sort(best_cor{:,:}, 'descend');
best_cor_desc = best_cor(:,idx);
figure()
heatmap(cor_matrix)

%}

%% Part 3

% 3a

% Create a normal linear regression model, using cross validation
% Store error and model pareameters
X_no_target = X_standardized(:,1:81);
target = X_standardized(:,82);

%Partition data into folds
f = 10;
cv = cvpartition(size(X_no_target,1), 'KFold', f);

% Define linear regression model
lm = fitlm(X_no_target, target, 'Intercept', true);

% Test the model on the k folds, store rms error and parameters
rms_error = zeros(f,1);
model_params = cell(f,1);

for k = 1:f

    % Set up training and testing set
    trainIdx = cv.training(k);  % indices for training set
    testIdx = cv.test(k);       % indices for test set
    X_train = X_no_target(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = X_no_target(testIdx,:);
    y_test = target(testIdx,:);

    % Fit linear model on training set
    lm_k = fitlm(X_train, y_train, 'Intercept', true);
    % Evaluate on test set
    y_pred = predict(lm_k, X_test);
    % Calculate RMS error
    rms_error(k) = sqrt(mean((y_test - y_pred).^2));
    % Store model parameters
    model_params{k} = lm_k.Coefficients;

end


%% Part 3b

% K-means wasn't good for our data so a stratified model doesn't make sense
% Since we have so many features, lets try removing some of the less
% important ones
% Choose features with least correlation to critical temperature to 
% remove first

% Lets try removing 10 to start
reduced_table = superconductor_data;
reduced_table = removevars(reduced_table, ["gmean_fie", "entropy_ThermalConductivity", ...
    "mean_fie", "mean_atomic_radius", "wtd_gmean_ElectronAffinity", "wtd_mean_ElectronAffinity", ...
    "mean_atomic_mass", "std_Density", "wtd_entropy_ThermalConductivity", "range_FusionHeat"]);
reduced_matrix = reduced_table{:,:};
reduced_standardized = normalize(reduced_matrix);
reduced_no_target = reduced_standardized(:,1:71);

% Perform linear regression on reduced data to see if it performs better
f = 10;
cv = cvpartition(size(reduced_no_target,1), 'KFold', f);
rms_error_red = zeros(f,1);
model_params_red = cell(f,1);

for k = 1:f

    trainIdx = cv.training(k);  % indices for training set
    testIdx = cv.test(k);       % indices for test set
    X_train = reduced_no_target(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = reduced_no_target(testIdx,:);
    y_test = target(testIdx,:);

    % Fit linear model on training set
    lm_k_red = fitlm(X_train, y_train, 'Intercept', true);
    % Evaluate on test set
    y_pred_red = predict(lm_k_red, X_test);
    % Calculate RMS error
    rms_error_red(k) = sqrt(mean((y_test - y_pred_red).^2));
    % Store model parameters
    model_params_red{k} = lm_k_red.Coefficients;

end

% It performed worse! Less features -> Less predicting power

% Lets try adding a few features instead
% Make a new table with no target feature
new_features_table = removevars(superconductor_data,"critical_temp");

% Need to shift so log doesnt end up complex
% Store correlations for newly logged features
newcorr = zeros(81,1);
for i = 1:81
    X_shift = X_standardized(:,i) - min(X_standardized(:,i)) + 1;
    newcorr(i) = corr(log10(X_shift),target);
end
newcorr = newcorr';
best_cormatrix = best_cor{:,:};
% Check corr of new feature vs corr of original feature
% Add the new feature for each one that improved
for i = 1:81
    if (abs(newcorr(1,i)) > best_cormatrix(1,i))

        X_shift = X_standardized(:,i) - min(X_standardized(:,i)) + 1;
        new_features_table.(num2str(i)) = log10(X_shift);
    end
end

% Perform and store corr of squared original features
newcorr = zeros(81,1);
for i = 1:81
    X_shift = X_standardized(:,i);
    newcorr(i) = corr((X_shift).^2,target);
end
newcorr = newcorr';
best_cormatrix = best_cor{:,:};
% Check corr of new squared feature vs corr of original feature
% Add the new feature for each one that improved
for i = 1:81
    if (abs(newcorr(1,i)) > best_cormatrix(1,i))
        X_shift = X_standardized(:,i);
        new_features_table.(num2str(i*10)) = (X_shift).^2;
    end
end
% Make table into a matrix and normalize
new_features_matrix = new_features_table{:,:};
new_features_matrix = normalize(new_features_matrix);

% Perform cross validation and test the model with newly added features
f = 10;
cv = cvpartition(size(new_features_matrix,1), 'KFold', f);
% Test the model on the k folds, store rms error
rms_error_new = zeros(f,1);
model_params_new = cell(f,1);
for k = 1:f
    trainIdx = cv.training(k);  % indices for training set
    testIdx = cv.test(k);       % indices for test set
    X_train = new_features_matrix(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = new_features_matrix(testIdx,:);
    y_test = target(testIdx,:);
    % Fit linear model on training set
    lm_k_new = fitlm(X_train, y_train, 'Intercept', true);
    % Evaluate on test set
    y_pred_new = predict(lm_k_new, X_test);
    % Calculate RMS error
    rms_error_new(k) = sqrt(mean((y_test - y_pred_new).^2));
    % Store model parameters
    model_params_new{k} = lm_k_new.Coefficients;
end
% Print performances of models from part 3a & 3b
fprintf('3a mean for normal lm: %f \n', mean(rms_error))
fprintf('3b mean for less features lm: %f \n', mean(rms_error_red))
fprintf('3b mean for extra features lm: %f \n', mean(rms_error_new))


%% Part 3c

% part 3.c.1

% Perform regularization on the model from 3b
% Split data into training and test sets
cv = cvpartition(size(new_features_matrix,1),'HoldOut',0.2);
idxTrain = training(cv);
Xtrain = new_features_matrix(idxTrain,:);
ytrain = target(idxTrain,:);
Xtest = new_features_matrix(~idxTrain,:);
ytest = target(~idxTrain,:);
% Define a range of lambda values to test
lambda_vals = logspace(-2, 5, 100);
% Initialize arrays to store RMS values for each lambda value
rms_train = zeros(length(lambda_vals),1);
rms_test = zeros(length(lambda_vals),1);
coeffs = zeros(length(lambda_vals), size(Xtrain,2));
% Loop over lambda values and fit regularized models
for i = 1:length(lambda_vals)
    lambda = lambda_vals(i);
    r_mdl = fitrlinear(Xtrain, ytrain, 'Regularization', 'ridge', 'Lambda', lambda);
    % Evaluate RMS on training and test sets
    yhat_train = predict(r_mdl, Xtrain);
    rms_train(i) = sqrt(mean((ytrain - yhat_train).^2));
    yhat_test = predict(r_mdl, Xtest);
    rms_test(i) = sqrt(mean((ytest - yhat_test).^2));
    coeffs(i,:) = r_mdl.Beta;
end
r_mdl = fitrlinear(Xtrain, ytrain, 'Regularization', 'ridge', 'Lambda', .1150);
% Extract the RMSE for the desired lambda value
rmse_for_lambda = rms_test(16);
fprintf('rms for regularized model: %f \n', rmse_for_lambda)
% Plot RMS vs lambda
figure()
semilogx(lambda_vals, rms_train, 'b-', 'DisplayName', 'Train');
hold on;
semilogx(lambda_vals, rms_test, 'r-', 'DisplayName', 'Test');
xlabel('\lambda');
ylabel('RMS');
legend();
title('\lambda vs RMS for initial regularized model')
% Plot regularization path
figure()
semilogx(lambda_vals, coeffs, '-');
xlabel('\lambda');
ylabel('Coefficient value');
title('Regularization path');
% Plot linear model fit on training and test data
figure()
scatter(Xtrain, ytrain, 'b', 'filled', 'DisplayName', '');
hold on;
scatter(Xtest, ytest, 'r', 'filled', 'DisplayName', '');
xvals = linspace(min(new_features_matrix(:)), max(new_features_matrix(:)), 100)';
xvals = repmat(xvals, 1, size(new_features_matrix, 2));
yvals_train = predict(r_mdl, xvals);
plot(xvals, yvals_train, 'k-', 'LineWidth', 2, 'DisplayName', 'Linear model fit');
xlabel('Feature value');
ylabel('Target value');
legend('Training data', 'Testing data', 'Linear model fit');
title('Linear model fit on training and testing data');


%for part 3.c.2

% Revise feature engineering to create a new regularized model
% We are dealing with underfitting for regularized linear model,
% So going to add randomly generated features to add regression power
% As per Prof suggestion and lec 9 notes
% Define the size of matrix B and vector v
k = 1000; % The number of additional features you want to create
B = randn(126, k);  % A random matrix of size [m, k]
v = randn(1, k);    % A random vector of size [1, k]

% Define the transformation function
% Only want positive values so use max func
f = @(x) max(0, x.*B + v);

X_new = zeros(size(new_features_matrix,1), size(B,2));
for i = 1:size(new_features_matrix,1)
    X_new(i,:) = max(0, new_features_matrix(i,:)*B + v);
end
% Add onto data set and normalize
new_features_matrix = horzcat(new_features_matrix, X_new);
new_features_matrix = normalize(new_features_matrix);
% Split into test and training sets
cv = cvpartition(size(new_features_matrix,1),'HoldOut',0.2);
idxTrain = training(cv);
Xtrain = new_features_matrix(idxTrain,:);
ytrain = target(idxTrain,:);
Xtest = new_features_matrix(~idxTrain,:);
ytest = target(~idxTrain,:);
% Define a range of lambda values to test
lambda_vals = logspace(-2, 5, 100);
% Initialize arrays to store RMS values for each lambda value
rms_train_rand = zeros(length(lambda_vals),1);
rms_test_rand = zeros(length(lambda_vals),1);
coeffs_rand = zeros(length(lambda_vals), size(Xtrain,2));
% Loop over lambda values and fit regularized models

for i = 1:length(lambda_vals)
    lambda = lambda_vals(i);
    r_mdl = fitrlinear(Xtrain, ytrain, 'Regularization', 'ridge', 'Lambda', lambda);
    % Evaluate RMS on training and test sets
    yhat_train = predict(r_mdl, Xtrain);
    rms_train_rand(i) = sqrt(mean((ytrain - yhat_train).^2));
    yhat_test = predict(r_mdl, Xtest);
    rms_test_rand(i) = sqrt(mean((ytest - yhat_test).^2));
    coeffs_rand(i,:) = r_mdl.Beta;
end
% Plot RMS vs lambda for revised regularized model
figure()
semilogx(lambda_vals, rms_train_rand, 'b-', 'DisplayName', 'Train');
hold on;
semilogx(lambda_vals, rms_test_rand, 'r-', 'DisplayName', 'Test');
xlabel('\lambda');
ylabel('RMS');
legend();
title('\lambda vs RMS for supplemented data regularized model')
% Define model for the best lambda, this will be used to plot
rand_mdl = fitrlinear(Xtrain, ytrain, 'Regularization', 'ridge', 'Lambda', .1150);
% Extract the RMSE for the desired lambda value
rmse_for_lambda_rand = rms_test_rand(3);
fprintf('rms for regularized model: %f \n', rmse_for_lambda_rand)
% Plot this stupid ass model
figure()
scatter(Xtrain, ytrain, 'b', 'filled', 'DisplayName', '');
hold on;
scatter(Xtest, ytest, 'r', 'filled', 'DisplayName', '');
xvals = linspace(min(new_features_matrix(:)), max(new_features_matrix(:)), 100)';
xvals = repmat(xvals, 1, size(new_features_matrix, 2));
yvals_train = predict(rand_mdl, xvals);
plot(xvals, yvals_train, 'k-', 'LineWidth', 2, 'DisplayName', 'Linear model fit');
xlabel('Feature value');
ylabel('Target value');
legend('Training data', 'Testing data', 'Linear model fit');
title('Linear model fit on training and testing data');

%% Part 3d

% Most of this code is recycled from Part 3a, we are only switching out
% the minimizing function from linear to non-linear everything else 
% stays the same

% Partition data into 10 folds for cross-validation
X_matrix = superconductor_data{:,:};
X_standardized = normalize(X_matrix);
X_no_target = X_standardized(:,1:81);
target = X_standardized(:,82);
folds = 10;
cv = cvpartition(size(X_no_target,1), 'KFold', folds);
rms_error = zeros(folds,1);
model_params = cell(folds,1);

% Remember to set this!! # of parameters you want to use
% Make sure to set the correct value or else it will complain (too small)
num_params = 3;

% Warning: Can take very long time
for k = 1:folds

    % Set up training and testing set for each fold
    trainIdx = cv.training(k); 
    testIdx = cv.test(k); 
    X_train = X_no_target(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = X_no_target(testIdx,:);
    y_test = target(testIdx,:);

    % Initial values for parameters, I just use ones for consistency,
    % can change to rand*constant instead
    x0 = ones(1, num_params);

    % Optional options to use levenberg-marquardt taught in class
    %options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
    %lb = [];
    %ub = [];

    % Non-linear fitting algorithm
    parameters = lsqcurvefit(@fun, x0, X_train, y_train);
    %parameters = lsqcurvefit(@fun, x0, X_train, y_train, lb, ub, options);
    
    % Predicted y value
    y_pred = fun(parameters, X_test);

    % Calculate RMS error
    rms_error(k) = sqrt(mean((y_test - y_pred).^2));

    % Store model parameters
    model_params_nl{k} = parameters;

end

disp(rms_error);



%% part 4
%part 4b
%find the std and mean of each row for column 1 of each table for each fold
%params for part 3a
%take column 1 from each table corresponding to the parameter estimates
%for each fold
% rename the column in each table
for i = 1:10
    model_params{i}.Properties.VariableNames{'Estimate'} = sprintf('Estimate_%d', i);
end
% Extract the first column of each table using cellfun
cols = cellfun(@(t) t(:, 1), model_params, 'UniformOutput', false);
% Concatenate the columns into a single table
result = cat(2, cols{:});
result = result(2:82,:);
% Compute the mean and standard deviation of each row
row_means_3a_params = mean(result{:,:}, 2);
row_stds_3a_params = std(result{:,:}, 0, 2);


%params for part 3b
%take column 1 from each table corresponding to the parameter estimates
%for each fold
% rename the column in each table
for i = 1:10
    model_params_new{i}.Properties.VariableNames{'Estimate'} = sprintf('Estimate_%d', i);
end
% Extract the first column of each table using cellfun
cols = cellfun(@(t) t(:, 1), model_params_new, 'UniformOutput', false);
% Concatenate the columns into a single table
result_3b = cat(2, cols{:});
result_3b = result_3b(2:126,:);
% Compute the mean and standard deviation of each row
row_means_3b_params = mean(result_3b{:,:}, 2);
row_stds_3b_params = std(result_3b{:,:}, 0, 2);

%params for part 3c
%just stored in coeffs and coeffs_rand
%for each model respectively
%lambda chosen for 3.c.1 was .1150 which corresponds to entry 16 so take
%16th row
%transpose for continuity
coeffs_3c1 = coeffs(16,:);
coeffs_3c1 = coeffs_3c1';
}
%Lambda chosen for 3.c.2 was also .1150
coeffs_3c2 = coeffs_rand(16,:);
coeffs_3c2 = coeffs_3c2';


%parameters for part 3d.
model_params_nl = model_params_nl';
model_params_nl = cellfun(@transpose,model_params_nl,'UniformOutput',false);
cols = cellfun(@(t) t(:, 1), model_params_nl, 'UniformOutput', false);
% Concatenate the columns into a single table
result_3d = cat(2, cols{:});
result_3d = result_3d(2:end,:);
% Compute the mean and standard deviation of each row
row_means_3d_params = mean(result_3d(:,:), 2);
row_stds_3d_params = std(result_3d(:,:), 0, 2);



%Moved function definitions to the end of script for functionality


% Non-linear prediction function for part 3d
% Minimize (fun(x, xdata(i)) - y(i))^2
% x is parameter vector, xdata(i) is feature vector for each data point i
function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)       
        for j = 1:size(xdata(i,:))
            F(i, 1) = F(i, 1) + abs(log(x(1)*xdata(i,j))) * x(2)^xdata(i,j);
        end
        F(i, 1) = F(i, 1) + x(3);
    end
end

% A collection of non-linear functions I've tried for part 3d. Just
% copy and paste in the one you want I guess

%{

% If I just use the formulation for linear least squares I get RMSE ~ 0.51
% which is basically the same as part 3a, as expected. 
% num_params = 81;

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = xdata(i,:)*transpose(x);
    end
end

% Non-linear Modification models:

% Terrible log fit, RMSE ~ 1.4
% num_params = 81;

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = log(abs(xdata(i,:)*transpose(x)));
    end
end


% Squaring isn't much better RMSE ~ 0.85
% num_params = 81;

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = (xdata(i,:)*transpose(x))^2;
    end
end


% Cubing does better but still not as good as regular linear
% num_params = 81, RMSE ~ 0.55

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = (xdata(i,:)*transpose(x))^3;
    end
end


% 4th power gives RMSE ~ 12.7 and takes long time


% Q function (tail area for standard gaussian), RMSE ~ 0.77
% num_params = 81, Takes forever

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = qfunc(xdata(i,:)*transpose(x));
    end
end

% Inverse tangent, RMSE ~ 0.52
% num_params - 81

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = atan(xdata(i,:)*transpose(x));
    end
end


Orignal non-linear models:

% RMSE ~ 1, num_params = 5

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = norm(x(1).*exp(sin(x(2).*xdata(i,:))).*xdata(i,:).^x(3)./x(4))/x(5);
    end
end


% RMSE ~ 1.2, num_params = 3

function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)       
        for j = 1:size(xdata(i,:))
            F(i, 1) = F(i, 1) + abs(log(x(1)*xdata(i,j))) * x(2)^xdata(i,j);
        end
        F(i, 1) = F(i, 1) + x(3);
    end
end

%}
