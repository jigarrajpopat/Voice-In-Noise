% downloadFolder = 'D:\MATLAB\2. CTME1 Final Project';
downloadFolder = 'E:\UM_hard_drive_folder\Course Material\1. Fall 2020\Current Trends in ME 1\2. CTME1 Final Project';

datasetFolder = fullfile(downloadFolder,'google_speech');
adsTrain = audioDatastore(fullfile(datasetFolder, 'train'), "Includesubfolders",true);
adsValidation = audioDatastore(fullfile(datasetFolder, 'validation'), "Includesubfolders",true);

[data,adsInfo] = read(adsTrain);
Fs = adsInfo.SampleRate;
sound(data,Fs);
timeVector = (1/Fs) * (0:numel(data)-1);
plot(timeVector,data)
ylabel("Amplitude")
xlabel("Time (s)")
title("Sample Audio")
grid on

win = hamming(50e-3 * Fs,'periodic');
detectSpeech(data,Fs,'Window',win);
speechIndices = detectSpeech(data,Fs,'Window',win);
sound(data(speechIndices(1,1):speechIndices(1,2)),Fs);

speechIndices(1,1) = max(speechIndices(1,1) - 5*numel(win),1);
speechIndices(1,2) = min(speechIndices(1,2) + 5*numel(win),numel(data));
sound(data(speechIndices(1,1):speechIndices(1,2)),Fs);

reset(adsTrain)
adsTrain = shuffle(adsTrain);
adsValidation = shuffle(adsValidation);

TM = [];
for index1 = 1:500
     data = read(adsTrain); 
     [~,T] = detectSpeech(data,Fs,'Window',win);
     TM = [TM;T];
end

T = mean(TM);

reset(adsTrain)

duration = 2000*Fs;
audioTraining = zeros(duration,1);

maskTraining = zeros(duration,1);

maxSilenceSegment = 2;

numSamples = 1;    
while numSamples < duration
    data = read(adsTrain);
    data = data ./ max(abs(data)); % Normalize amplitude

    % Determine regions of speech
    idx = detectSpeech(data,Fs,'Window',win,'Thresholds',T);

    % If a region of speech is detected
    if ~isempty(idx)
        
        % Extend the indices by five frames
        idx(1,1) = max(1,idx(1,1) - 5*numel(win));
        idx(1,2) = min(length(data),idx(1,2) + 5*numel(win));
        
        % Isolate the speech
        data = data(idx(1,1):idx(1,2));
        
        % Write speech segment to training signal
        audioTraining(numSamples:numSamples+numel(data)-1) = data;
        
        % Set VAD baseline
        maskTraining(numSamples:numSamples+numel(data)-1) = true;
        
        % Random silence period
        numSilenceSamples = randi(maxSilenceSegment*Fs,1,1);
        numSamples = numSamples + numel(data) + numSilenceSamples;
    end
end

figure
range = 1:10*Fs;
plot((1/Fs)*(range-1),audioTraining(range));
hold on
plot((1/Fs)*(range-1),maskTraining(range));
grid on
lines = findall(gcf,"Type","Line");
lines(1).LineWidth = 2;
xlabel("Time (s)")
legend("Signal","Speech Region")
title("Training Signal (first 10 seconds)");

sound(audioTraining(range),Fs);

noise = audioread("WashingMachine-16-8-mono-1000secs.wav");
noise = resample(noise,2,1);

audioTraining = audioTraining(1:numel(noise));
SNR = -10;
noise = 10^(-SNR/20) * noise * norm(audioTraining) / norm(noise);
audioTrainingNoisy = audioTraining + noise; 
audioTrainingNoisy = audioTrainingNoisy / max(abs(audioTrainingNoisy));

figure
plot((1/Fs)*(range-1),audioTrainingNoisy(range));
hold on
plot((1/Fs)*(range-1),maskTraining(range));
grid on
lines = findall(gcf,"Type","Line");
lines(1).LineWidth = 2;
xlabel("Time (s)")
legend("Noisy Signal","Speech Area")
title("Training Signal (first 10 seconds)");

sound(audioTrainingNoisy(range),Fs);

speechIndices = detectSpeech(audioTrainingNoisy,Fs,'Window',win);
speechIndices(:,1) = max(1,speechIndices(:,1) - 5*numel(win));
speechIndices(:,2) = min(numel(audioTrainingNoisy),speechIndices(:,2) + 5*numel(win));
noisyMask = zeros(size(audioTrainingNoisy));
for ii = 1:size(speechIndices)
    noisyMask(speechIndices(ii,1):speechIndices(ii,2)) = 1;
end

figure
plot((1/Fs)*(range-1),audioTrainingNoisy(range));
hold on
plot((1/Fs)*(range-1),noisyMask(range));
grid on
lines = findall(gcf,"Type","Line");
lines(1).LineWidth = 2;
xlabel("Time (s)")
legend("Noisy Signal","Mask from Noisy Signal")
title("Training Signal (first 10 seconds)");

duration = 200*Fs;
audioValidation = zeros(duration,1);
maskValidation = zeros(duration,1);

numSamples = 1;    
while numSamples < duration
    data = read(adsValidation);
    data = data ./ max(abs(data)); % Normalize amplitude
    
    % Determine regions of speech
    idx = detectSpeech(data,Fs,'Window',win,'Thresholds',T);
    
    % If a region of speech is detected
    if ~isempty(idx)
        
        % Extend the indices by five frames
        idx(1,1) = max(1,idx(1,1) - 5*numel(win));
        idx(1,2) = min(length(data),idx(1,2) + 5*numel(win));

        % Isolate the speech
        data = data(idx(1,1):idx(1,2));
        
        % Write speech segment to training signal
        audioValidation(numSamples:numSamples+numel(data)-1) = data;
        
        % Set VAD Baseline
        maskValidation(numSamples:numSamples+numel(data)-1) = true;
        
        % Random silence period
        numSilenceSamples = randi(maxSilenceSegment*Fs,1,1);
        numSamples = numSamples + numel(data) + numSilenceSamples;
    end
end

noise = audioread("WashingMachine-16-8-mono-200secs.wav");
noise = resample(noise,2,1);
noise = noise(1:duration);
audioValidation = audioValidation(1:numel(noise));

noise = 10^(-SNR/20) * noise * norm(audioValidation) / norm(noise);
audioValidationNoisy = audioValidation + noise; 
audioValidationNoisy = audioValidationNoisy / max(abs(audioValidationNoisy));


%ML part
afe = audioFeatureExtractor('SampleRate',Fs, ...
    'Window',hann(256,"Periodic"), ...
    'OverlapLength',128, ...
    ...
    'spectralCentroid',true, ...
    'spectralCrest',true, ...
    'spectralEntropy',true, ...
    'spectralFlux',true, ...
    'spectralKurtosis',true, ...
    'spectralRolloffPoint',true, ...
    'spectralSkewness',true, ...
    'spectralSlope',true, ...
    'harmonicRatio',true);
featuresTraining = extract(afe,audioTrainingNoisy);

[numWindows,numFeatures] = size(featuresTraining);

M = mean(featuresTraining,1);
S = std(featuresTraining,[],1);
featuresTraining = (featuresTraining - M) ./ S;

featuresValidation = extract(afe,audioValidationNoisy);
featuresValidation = (featuresValidation - mean(featuresValidation,1)) ./ std(featuresValidation,[],1);

windowLength = numel(afe.Window);
hopLength = windowLength - afe.OverlapLength;
range = (hopLength) * (1:size(featuresTraining,1)) + hopLength;
maskMode = zeros(size(range));
for index = 1:numel(range)
    maskMode(index) = mode(maskTraining( (index-1)*hopLength+1:(index-1)*hopLength+windowLength ));
end
maskTraining = maskMode.';
maskTrainingCat = categorical(maskTraining);

range = (hopLength) * (1:size(featuresValidation,1)) + hopLength;
maskMode = zeros(size(range));
for index = 1:numel(range)
    maskMode(index) = mode(maskValidation( (index-1)*hopLength+1:(index-1)*hopLength+windowLength ));
end
maskValidation = maskMode.';
maskValidationCat = categorical(maskValidation);

sequenceLength = 800;
sequenceOverlap = round(0.75*sequenceLength);
trainFeatureCell = helperFeatureVector2Sequence(featuresTraining',sequenceLength,sequenceOverlap);
trainLabelCell = helperFeatureVector2Sequence(maskTrainingCat',sequenceLength,sequenceOverlap);


layers2 = [ ...    
    sequenceInputLayer( size(featuresValidation,2) )    
    bilstmLayer(200,"OutputMode","sequence")   
    %bilstmLayer(200,"OutputMode","sequence")   
    fullyConnectedLayer(2)   
    softmaxLayer   
    classificationLayer      
    ];

maxEpochs = 20;
miniBatchSize = 64;
options = trainingOptions("adam", ...
    "MaxEpochs",maxEpochs, ...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Verbose",0, ...
    "SequenceLength",sequenceLength, ...
    "ValidationFrequency",floor(numel(trainFeatureCell)/miniBatchSize), ...
    "ValidationData",{featuresValidation.',maskValidationCat.'}, ...
    "Plots","training-progress", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",5);

%training
doTraining = true;
if doTraining
   [speechDetectNet,netInfo] = trainNetwork(trainFeatureCell,trainLabelCell,layers2,options);
    fprintf("Validation accuracy: %f percent.\n", netInfo.FinalValidationAccuracy);
else
    load speechDetectNet
end
%

%validate
EstimatedVADMask = classify(speechDetectNet,featuresValidation.');
EstimatedVADMask = double(EstimatedVADMask);
EstimatedVADMask = EstimatedVADMask.' - 1;

figure
cm = confusionchart(maskValidation,EstimatedVADMask,"title","Validation Accuracy");
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";

tp = sum((maskValidation==1)&(EstimatedVADMask==1));
fp = sum((maskValidation==0)&(EstimatedVADMask==1));
fn = sum((maskValidation==1)&(EstimatedVADMask==0));
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1 = (2*precision*recall)/(precision+recall);
disp(f1);

%-------------------------------------test------------------------------------%

adsTest = audioDatastore(fullfile(datasetFolder, 'test'), "Includesubfolders",true);

duration = 200*Fs;
audioTest = zeros(duration,1);
maskTest = zeros(duration,1);

numSamples = 1;    
while numSamples < duration
    data = read(adsTest);
    data = data ./ max(abs(data)); % Normalize amplitude
    
    % Determine regions of speech
    idx = detectSpeech(data,Fs,'Window',win,'Thresholds',T);
    
    % If a region of speech is detected
    if ~isempty(idx)
        
        % Extend the indices by five frames
        idx(1,1) = max(1,idx(1,1) - 5*numel(win));
        idx(1,2) = min(length(data),idx(1,2) + 5*numel(win));

        % Isolate the speech
        data = data(idx(1,1):idx(1,2));
        
        % Write speech segment to training signal
        audioTest(numSamples:numSamples+numel(data)-1) = data;
        
        % Set VAD Baseline
        maskTest(numSamples:numSamples+numel(data)-1) = true;
        
        % Random silence period
        numSilenceSamples = randi(maxSilenceSegment*Fs,1,1);
        numSamples = numSamples + numel(data) + numSilenceSamples;
    end
end

noise = audioread("WashingMachine-16-8-mono-200secs.wav");
noise = resample(noise,2,1);
noise = noise(1:duration);
audioTest = audioTest(1:numel(noise));

noise = 10^(-SNR/20) * noise * norm(audioTest) / norm(noise);
audioTestNoisy = audioTest + noise; 
audioTestNoisy = audioTestNoisy / max(abs(audioTestNoisy));


featuresTest = extract(afe,audioTestNoisy);
featuresTest = (featuresTest - mean(featuresTest,1)) ./ std(featuresTest,[],1);

windowLength = numel(afe.Window);
hopLength = windowLength - afe.OverlapLength;
range = (hopLength) * (1:size(featuresTest,1)) + hopLength;
maskMode = zeros(size(range));
for index = 1:numel(range)
    maskMode(index) = mode(maskTest( (index-1)*hopLength+1:(index-1)*hopLength+windowLength ));
end
maskTest = maskMode.';
maskTestCat = categorical(maskTest);

%test it out
EstimatedVADMask2 = classify(speechDetectNet,featuresTest.');
EstimatedVADMask2 = double(EstimatedVADMask2);
EstimatedVADMask2 = EstimatedVADMask2.' - 1;

figure
cm = confusionchart(maskTest,EstimatedVADMask2,"title","Test Accuracy");
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";

tp = sum((maskTest==1)&(EstimatedVADMask2==1));
fp = sum((maskTest==0)&(EstimatedVADMask2==1));
fn = sum((maskTest==1)&(EstimatedVADMask2==0));
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1_test = (2*precision*recall)/(precision+recall);
disp(f1_test);



