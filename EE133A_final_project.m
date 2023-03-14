clear; clc; close all;
%%Part 1
%element_data is a breakdown of each superconductor into its chemical
%formula by element, also listing its critical temp, 
%and the written chemical formula
element_data = readtable("unique_m.csv");
%superconductor_data gives the feature values of the same superconductors
%that are in the element_data
superconductor_data = readtable("train.csv");

%comment out for speed when doing other parts of project
%{
%%Just looking at some basic plots and info from our tables, to grasp info
%Histogram of the critical temps from first table
histogram(element_data.("critical_temp")(1:21263))
figure()
%histogram of critical temps from second table to confirm they are the same
histogram(superconductor_data.("critical_temp")(1:21263))
figure()


%plot the percentage of superconductors with each element
x = 1:86;
elements = element_data(:,1:86);
for i = 1:86
    table(i) = (sum(elements.(i) ~= 0))/21263;
end

%store the table as an array so we can associate each value with its
%element
table = array2table(table);
table.Properties.VariableNames = elements.Properties.VariableNames;
[~, idx] = sort(table{:,:}, 'descend');
table = table(:,idx);
%plotting stuff
scatter(x,table{:,:});
text(x,table{:,:},table.Properties.VariableNames)
xlim([0 86])

%find the standard deviation and mean of critical temp for each element
%whereever there is a non-zero entry, add the corresponding crit temp
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
%put the arrays into tables and sort them in descending order
means = array2table(means);
stds = array2table(stds);
means.Properties.VariableNames = elements.Properties.VariableNames;
stds.Properties.VariableNames = elements.Properties.VariableNames;
[~, idx] = sort(means{:,:}, 'descend');
means = means(:,idx);
[~, idx] = sort(stds{:,:}, 'descend');
stds = stds(:,idx);
%plot the mean and std
figure()
scatter(x,means{:,:});
text(x, means{:,:}, means.Properties.VariableNames)
figure()
scatter(x,stds{:,:});
text(x, stds{:,:}, stds.Properties.VariableNames)

%plot the mean vs std both linear and log
figure()
scatter(means{:,:}, stds{:,:});
figure()
scatter(log(means{:,:}), stds{:,:})
%}
%% part 2
%create an unlabeled matrix to perform standardization
%find mean and std of each feature
X_matrix = superconductor_data{:,:};
X_standardized = normalize(X_matrix);
mean_features = varfun(@mean, superconductor_data, 'InputVariables', @isnumeric);
std_features = varfun(@std, superconductor_data, 'InputVariables', @isnumeric);
%perform k-means clustering
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

%% Part 3
%3a
X_no_target = X_standardized(:,1:81);
target = X_standardized(:,82);
%partition data into folds without target column
f = 10;
cv = cvpartition(size(X_no_target,1), 'KFold', f);
%define linear regression model
lm = fitlm(X_no_target, target, 'Intercept', true);
%test the model on the k folds, store rms error
rms_error = zeros(f,1);
model_params = cell(f,1);
for k = 1:f
    trainIdx = cv.training(k); % indices for training set
    testIdx = cv.test(k); % indices for test set
    X_train = X_no_target(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = X_no_target(testIdx,:);
    y_test = target(testIdx,:);
    %fit linear model on training set
    lm_k = fitlm(X_train, y_train, 'Intercept', true);
    %evaluate on test set
    y_pred = predict(lm_k, X_test);
    %calculate RMS error
    rms_error(k) = sqrt(mean((y_test - y_pred).^2));
    %store model parameters
    model_params{k} = lm_k.Coefficients;
end

%3b
%K-means wasn't good for our data so a stratified model doesn't make sense
%Since we have so many features, lets try removing some of the less
%important ones
%I choose which features to remove based on their correlation to the
%critical temperature
%Lets try removing 10 to start
reduced_table = superconductor_data;
reduced_table = removevars(reduced_table, ["gmean_fie", "entropy_ThermalConductivity", ...
    "mean_fie", "mean_atomic_radius", "wtd_gmean_ElectronAffinity", "wtd_mean_ElectronAffinity", ...
    "mean_atomic_mass", "std_Density", "wtd_entropy_ThermalConductivity", "range_FusionHeat"]);
reduced_matrix = reduced_table{:,:};
reduced_standardized = normalize(reduced_matrix);
reduced_no_target = reduced_standardized(:,1:71);
%lets perform linear regression on reduced data to see if it performs
%better
f = 10;
cv = cvpartition(size(reduced_no_target,1), 'KFold', f);
%test the model on the k folds, store rms error
rms_error_red = zeros(f,1);
model_params_red = cell(f,1);
for k = 1:f
    trainIdx = cv.training(k); % indices for training set
    testIdx = cv.test(k); % indices for test set
    X_train = reduced_no_target(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = reduced_no_target(testIdx,:);
    y_test = target(testIdx,:);
    %fit linear model on training set
    lm_k_red = fitlm(X_train, y_train, 'Intercept', true);
    %evaluate on test set
    y_pred_red = predict(lm_k_red, X_test);
    %calculate RMS error
    rms_error_red(k) = sqrt(mean((y_test - y_pred_red).^2));
    %store model parameters
    model_params_red{k} = lm_k_red.Coefficients;
end
%it performed worse!
%Lets try adding a few features instead
%make a new table with no target feature
new_features_table = removevars(superconductor_data,"critical_temp");
%need to shift so log doesnt end up complex
%check which correlations improve when log10()ed
newcorr = zeros(81,1);
for i = 1:81
    X_shift = X_standardized(:,i) - min(X_standardized(:,i)) + 1;
    newcorr(i) = corr(log10(X_shift),target);
end
newcorr = newcorr';
best_cormatrix = best_cor{:,:};
%add a new feature for each feature that improved when log10()ed
for i = 1:81
    if (abs(newcorr(1,i)) > best_cormatrix(1,i))

        X_shift = X_standardized(:,i) - min(X_standardized(:,i)) + 1;
        new_features_table.(num2str(i)) = log10(X_shift);
    end
end
%check square
newcorr = zeros(81,1);
for i = 1:81
    X_shift = X_standardized(:,i);
    newcorr(i) = corr((X_shift).^2,target);
end
newcorr = newcorr';
best_cormatrix = best_cor{:,:};
%add a new feature for each feature that improved when squared
for i = 1:81
    if (abs(newcorr(1,i)) > best_cormatrix(1,i))
        X_shift = X_standardized(:,i);
        new_features_table.(num2str(i*10)) = (X_shift).^2;
    end
end
%make table into a matrix
new_features_matrix = new_features_table{:,:};
new_features_matrix = normalize(new_features_matrix);
%check corr
%test the data with extra features
f = 10;
cv = cvpartition(size(new_features_matrix,1), 'KFold', f);
%test the model on the k folds, store rms error
rms_error_new = zeros(f,1);
model_params_new = cell(f,1);
for k = 1:f
    trainIdx = cv.training(k); % indices for training set
    testIdx = cv.test(k); % indices for test set
    X_train = new_features_matrix(trainIdx,:);
    y_train = target(trainIdx,:);
    X_test = new_features_matrix(testIdx,:);
    y_test = target(testIdx,:);
    %fit linear model on training set
    lm_k_new = fitlm(X_train, y_train, 'Intercept', true);
    %evaluate on test set
    y_pred_new = predict(lm_k_new, X_test);
    %calculate RMS error
    rms_error_new(k) = sqrt(mean((y_test - y_pred_new).^2));
    %store model parameters
    model_params_new{k} = lm_k_new.Coefficients;
end

fprintf('mean for normal lm: %f \n', mean(rms_error))
fprintf('mean for less features lm: %f \n', mean(rms_error_red))
fprintf('mean for extra features lm: %f \n', mean(rms_error_new))

%3c
%find best lambda
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

%we are dealing with underfitting for regularized linear model,
%so going to add randomly generated features
%as per Prof suggestion and lec 9 notes
% Define the size of matrix B and vector v
k = 81; % The number of additional features you want to create
B = randn(126, k); % A random matrix of size [m, k]
v = randn(1, k); % A random vector of size [1, k]

% Define the transformation function
%only want positive values
f = @(x) max(0, x.*B + v);

% Alternatively, you can use the following loop to apply the transformation row by row
X_new = zeros(size(new_features_matrix,1), size(B,2));
for i = 1:size(new_features_matrix,1)
    X_new(i,:) = max(0, new_features_matrix(i,:)*B + v);
end
%add onto data set
new_features_matrix = horzcat(new_features_matrix, X_new);
new_features_matrix = normalize(new_features_matrix);
%test the performance of the new dataset
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
% Plot RMS vs lambda
figure()
semilogx(lambda_vals, rms_train_rand, 'b-', 'DisplayName', 'Train');
hold on;
semilogx(lambda_vals, rms_test_rand, 'r-', 'DisplayName', 'Test');
xlabel('\lambda');
ylabel('RMS');
legend();
title('\lambda vs RMS for supplemented data regularized model')
%define model for the best lambda, this will be used to plot
rand_mdl = fitrlinear(Xtrain, ytrain, 'Regularization', 'ridge', 'Lambda', .0138);
% Extract the RMSE for the desired lambda value
rmse_for_lambda_rand = rms_test_rand(3);
fprintf('rms for regularized model: %f \n', rmse_for_lambda_rand)
%plot this stupid ass model
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
