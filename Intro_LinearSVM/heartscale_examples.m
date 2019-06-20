%% Intro to SVM, SASQUATCH
% Description: basic implementation of LIBSVM using heart_scale dataset (provided in LIBSVM package
% download and outlined in README)

cd /home/mflounders/libsvm-3.23/matlab

%% Example 1

%load data 
[heart_scale_label, heart_scale_inst] = libsvmread('../heart_scale');
% train 
model = svmtrain(heart_scale_label, heart_scale_inst, '-t 0 -c 1');
% test
[predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, model); % test the training data

%% Example 2
%load data 
[heart_scale_label, heart_scale_inst] = libsvmread('../heart_scale');

% preset, for easy cross-validation
odd_data=heart_scale_inst(1:2:270,:);
even_data=heart_scale_inst(2:2:270,:);

odd_labels=heart_scale_label(1:2:270,:);
even_labels=heart_scale_label(2:2:270,:);

%%%%%%%%%%FOLD 1
% train
model = svmtrain(even_labels, even_data, '-t 0 -c 1');
% test
[predict_label1, accuracy1, dec_values1] = svmpredict(odd_labels, odd_data, model);
%%%%%%%%%%FOLD 2
% train
model = svmtrain(odd_labels, odd_data, '-t 0 -c 1');
% test
[predict_label2, accuracy2, dec_values2] = svmpredict(even_labels, even_data, model);

% calculate cross-validated accuracy
final_accuracy=(accuracy1 + accuracy2)/2;

