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
