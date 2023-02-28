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
%perform k-means for values of k 1-10
k_values = 1:10;
sse = zeros(size(k_values));
for i = 1:length(k_values)
    [~, centroids, sumd] = kmeans(X_standardized(:,1:81), k_values(i));
    sse(i) = sum(sumd);
end
%plot the elbow curve of the results to determine optimum k
figure();
plot(k_values, sse, 'bx-');
xlabel('Number of clusters');
ylabel('Sum of squared distances');
title('Elbow Curve');
%determine k mathematically by calculating where the elbow curve slope
%flattens out
diff_sse = diff(sse);
[~, optimal_k] = max(diff_sse);
optimal_k = optimal_k + 1;

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
