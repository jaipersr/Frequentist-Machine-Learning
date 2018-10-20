% Ryan Jaipersaud
% 9/29/2018
% https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
% The above link is to a data set that contains parameters associated with 
% a risk of heart disease. Factors incluse sex, age, weigth and other
% things.
% The output is on a scale from 0 (no heart disease) to 4.
% The code below executes three types of regression least squares, ridge
% and lasso.

clc;
clear all;

O = importfile1('processed.cleveland.data.csv'); % 303 rows of data points
O = (O-mean(O,1))./std(O); % This is to normalize the data 
size(O);

% This creates the input and output matrices
one = ones(250,1);
Xtrain = [one O(1:250,1:13)];
one2 = ones(25,1);
Xtest = [one2 O(251:275,1:13)];
one3 = ones(28,1);
Xvalid = [one3 O(276:303,1:13)];
ytrain = O(1:250,14); % training outputs
ytest = O(251:275,14); % test outputs
yvalid = O(276:303,14); % validation outputs

%------------------------Least Squares Regression Start------------------------%
Blsrt = inv( (transpose(Xtrain)*Xtrain) ) * transpose(Xtrain) * ytrain; % closed form solution for least squares
ylsrtpred1 = Xtrain * Blsrt;
ylsrtpred2 = Xtest * Blsrt; 

MSELSRT_train = ytrain - ylsrtpred1;
MSELSRT_train_error = mean(MSELSRT_train.^2,1);

MSELSRT_test = ytest - ylsrtpred2;
MSELSRT_test_error = mean(MSELSRT_test.^2,1) % Test Error to be displayed

%------------------------Least Squares Regression End------------------------%


%------------------------Ridge Regression Start -------------------------%

% The below code is meant is tune the paramater lambda and uses the
% validation set
Xvalid  = O(276:303,1:13);
yvalid = O(276:303,14);
I = eye(13,13);
lambda_error_combo = zeros(10,3);
j = 1; % index to move through lambda_error combo matrix
for k = 0.1 :0.1: 1.1
    lambda = k;
    BRidge = inv( (transpose(Xvalid)*Xvalid) + lambda*I ) * transpose(Xvalid)*yvalid;
    yRidgevalid = Xvalid*BRidge;
    MSERidgevalid = yvalid - yRidgevalid; 
    MSERidge_valid_Error = mean(MSERidgevalid.^2,1);
    
    lambda_error_combo(j,1) = lambda;
    lambda_error_combo(j,2) = MSERidge_valid_Error; 
    lambda_error_combo(j,3) = trace(Xvalid*inv((transpose(Xvalid)*Xvalid + lambda*I))*transpose(Xvalid)); % df(lambda)
    j = j+1;
end

% Min is a vector containing the min values of each column
% Index contain the row position of the min values
[Min,Index] = min(lambda_error_combo);

% lambda was found to be 0.1
lambda = lambda_error_combo(Index(1,2),1); % this finds the lambda with the lowest Error on the validation set

Xtrain  = O(1:250,1:13);
ytrain = O(1:250,14);
BRidge = inv( (transpose(Xtrain)*Xtrain) + lambda*I ) * transpose(Xtrain)*ytrain; % closed form solution for ridge resgression
yRidgepred1 = Xtrain * BRidge;
Xtest  = O(251:275,1:13);
ytest = O(251:275,14);
yRidgepred2 = Xtest * BRidge;


MSERidge_train = ytrain - yRidgepred1;
MSERidge_train_error = mean(MSERidge_train.^2,1);

MSERidge_test = ytest - yRidgepred2;
MSERidge_test_error = mean(MSERidge_test.^2,1) % Test Error to be displayed


%----------------------------Ridge Regression End--------------------------

%----------------------------Lasso Regression Start----------------------%
B = lasso(Xtrain,ytrain,'CV',10);
%transpose(B) % perhaps easier to look at than B

lassoPlot(B,'PredictorNames',{'age', 'sex' ,'cp', 'trestbps','chol', 'fbs', 'restecg' ,'thalach', 'exang','oldpeak','slope','ca','thal'});
legend('show','Location','NorthEast') % Show legend

% I drew a line on the lasso plot at L1 = 0.33544 and took the values at
% that point
% the non zero coefficents correspond to 
% 9 cp: chest pain type 
%-- Value 1: typical angina 
%-- Value 2: atypical angina 
%-- Value 3: non-anginal pain 
%-- Value 4: asymptomatic 
% oldpeak: ST depression induced by exercise relative to rest 
% ca: number of major vessels (0-3) colored by flourosopy
% thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
BLasso = transpose([0,0, 0.0081324,0,0,0,0,0,0,0.072435,0,0.11184,0.14303]);
yLassopred2 = Xtest * BLasso;
MSELasso_test = ytest - yLassopred2;
MSELasso_test_error = mean(MSELasso_test.^2,1)


yLassopred1 = Xtrain * BLasso;
MSELasso_train = ytrain - yLassopred1;
MSELasso_train_error = mean(MSELasso_train.^2,1);

fprintf('The lasso regression has the lowest error of all three regression at 0.7033.')
fprintf('The features selected were chest pain, oldpeak, ca, and thal.')
fprintf('Things such as age and sex were found to not matter as much.')

%----------------------------Lasso Regression End----------------------%




%-----------------------------File Processing---------------------------%

function processed = importfile1(filename, startRow, endRow)
%IMPORTFILE1 Import numeric data from a text file as a matrix.
%   PROCESSED = IMPORTFILE1(FILENAME) Reads data from text file FILENAME
%   for the default selection.
%
%   PROCESSED = IMPORTFILE1(FILENAME, STARTROW, ENDROW) Reads data from
%   rows STARTROW through ENDROW of text file FILENAME.
%
% Example:
%   processed = importfile1('processed.cleveland.data.csv', 1, 303);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2018/09/27 13:34:29

%% Initialize variables.
delimiter = ',';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% Read columns of data as text:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric text to numbers.
% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end


%% Replace non-numeric cells with 0.0
R = cellfun(@(x) (~isnumeric(x) && ~islogical(x)) || isnan(x),raw); % Find non-numeric cells
raw(R) = {0.0}; % Replace non-numeric cells

%% Create output variable
processed = cell2mat(raw);
end
