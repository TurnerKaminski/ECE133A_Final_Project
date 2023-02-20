clear; clc; close all;

%%Part 1
%data is already clean of mistakes/outliers from the UCI posting
%element_data is a breakdown of each superconductor into its chemical
%formula by element, also listing its critical temp, 
%and the written chemical formula
element_data = readtable("unique_m.csv");
%superconductor_data gives the feature values of the same superconductors
%that are in the element_data
superconductor_data = readtable("train.csv");


%%moving on to non required work
%Just looking at some basic plots and info from our tables, to grasp info
%Histogram of the critical temps from first table
histogram(element_data.("critical_temp")(1:21263))
figure()
%histogram of critical temps from second table to confirm they are the same
histogram(superconductor_data.("critical_temp")(1:21263))
figure()


%plot the percentage of superconductors with each element present within them
x = 1:86;
elements = element_data(:,1:86);
for i = 1:86
    table(i) = (sum(elements.(i) ~= 0))/21263;
end

%store the table as an array so we can associate each value with its
%element and then do classic plotting and labeling
table = array2table(table);
table.Properties.VariableNames = elements.Properties.VariableNames;
[~, idx] = sort(table{:,:}, 'descend');
table = table(:,idx);
scatter(x,table{:,:});
text(x,table{:,:},table.Properties.VariableNames)
xlim([0 86])

%group the critical temps by element
%whereever there is a non-zero entry, add the corresponding crit temp
%and then find the mean and store it for that corresponding element
for i = 1:86
    temp = 0;
    non_zero_entry_indices = find(elements.(i));
    if(length(non_zero_entry_indices) ~=0)
        for t = 1:length(non_zero_entry_indices)
            index_value = non_zero_entry_indices(t);
            temp(t) = element_data.("critical_temp")(index_value);
        end
    end
    means(i) = mean(temp);
end
%make it into a table, sort and plot
means = array2table(means);
means.Properties.VariableNames = elements.Properties.VariableNames;
[~, idx] = sort(means{:,:}, 'descend');
means = means(:,idx);
figure()
scatter(x,means{:,:});
text(x, means{:,:}, means.Properties.VariableNames)
