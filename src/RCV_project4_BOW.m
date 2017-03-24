%run('/Users/e.kim4/Documents/MATLAB/vlfeat-0.9.20/toolbox/vl_setup')
%vl_version verbose
%% 1: Bag Of Words
clear;
close all;
clc;

imgDir = '/Users/e.kim4/Documents/MATLAB/RCV_project4/vision_dataset';
imds = imageDatastore(imgDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%split into testing vs train images: split is 5,5
[testImages,trainImages] = splitEachLabel(imds,5);
%label the trainImages 1~10
trainlabels = grp2idx(trainImages.Labels);
sizelabels = size(trainlabels);
concatDesTrain = [];

for i = 1:sizelabels(1)
    train{i} = imread(trainImages.Files{i});
    test{i} = imread(testImages.Files{i});
    %read each train images % test images
    if size(train{i},3) ~= 3
        train{i} = imresize(train{i},[300,300]);
    elseif size(train{i},3) ~= 3
        test{i} = imresize(test{i},[300,300]);
    else
        train{i} = rgb2gray(imresize(train{i},[300,300]));
        test{i} = rgb2gray(imresize(test{i},[300,300]));
    end

    %format them as single (just to run it on vl_sift)
    %singleTrain{i} = single(train{i});
    %singleTest{i} = single(test{i});
    
    %For a subset of all the interstpoints in training image, clustering
    %the descriptors using k-means clustering:
    %output as k-visual words with each word has an associated 128x1
    %centroids 
    %%%USING VL_SIFT: find interest points and descriptor for testing and train images
    %[interestTrain{i}, desTrain{i}] = vl_sift(singleTrain{i});
    %[interestTest{i}, desTest{i}] = vl_sift(singleTest{i});
    
    %USING MATLAB BUILT IN FUNCTION
    interestTrain{i} = detectSURFFeatures(train{i});
    %getOnly 150 strongest interestpoint to compute
    interestTrain{i} = interestTrain{i}.selectStrongest(100);
    [desTrain{i}, validPTrain{i}] = extractFeatures(train{i}, interestTrain{i},'SURFSize',128);
    
    interestTest{i} = detectSURFFeatures(test{i});
    interestTest{i} = interestTest{i}.selectStrongest(100);
    [desTest{i}, validPTest{i}] = extractFeatures(test{i}, interestTest{i},'SURFSize',128);
    
    %put all the descriptors in one matrix for testing and train images
    concatDesTrain = [concatDesTrain desTrain{i}'];
    %concatDesTest = [concatDesTest desTest{i}];    
end;
%% For K=300 clustering
K = 300;
%so in here, assignments = closest centroids for each of Descriptor.
[centroid, assignTrain] = vl_kmeans(double(concatDesTrain), K); 

%seperate each assigned descriptors into each classes.
for i = 1:sizelabels(1)
    [closestToCentroidsTrain{i}, distanceTrain{i}] = knnsearch(centroid',desTrain{i});
end

for i = 1:sizelabels(1)
    [closestToCentroidsTest{i}, distanceTest{i}] = knnsearch(centroid',desTest{i});
end

concatFVTrain = [];
for i = 1:sizelabels(1)
    featureVectorTrain{i} = zeros(1,K);
    featureVectorTest{i} = zeros(1,K);
    %can be either size of Test or Train
    sizeDescript = size(closestToCentroidsTest{i});
    for j=1:sizeDescript(1)
        featureVectorTrain{i}(1,closestToCentroidsTrain{i}(j,1)) = featureVectorTrain{i}(1,closestToCentroidsTrain{i}(j,1)) + 1; 
        featureVectorTest{i}(1,closestToCentroidsTest{i}(j,1)) = featureVectorTest{i}(1,closestToCentroidsTest{i}(j,1)) + 1; 
    end
    % concatFVTrain's each row contains histogram of eachImages
    concatFVTrain(i,:) = [featureVectorTrain{i}];
    concatFVTest(i,:) = [featureVectorTest{i}];
end
%%show similar histograms
figure(1)
    for i = 1:4
        if i <= 2
            subplot(2,2,i)
            bar(featureVectorTrain{45+i});
            %featureVectorTest{trackTrain} = dummyTrain.Values;
            hold on;
            axis([0 K+1 0 15])
            title(sprintf('Similarity: class %d Train image %d: Zebra',10,45+i));
            hold off;
        else
            subplot(2,2,i)
            bar(featureVectorTest{45+i});
            %featureVectorTest{trackTest} = dummyTest.Values;
            hold on;
            axis([0 K+1 0 15])
            title(sprintf('Similarity: class %d Test image %d: Zebra',10,45+i));
            hold off;
        end
        
    end
    
%%show different histogram
figure(2)
    for i = 1:4
        if i <= 2
            subplot(2,2,i)
            bar(featureVectorTrain{45+i});
            %featureVectorTest{trackTrain} = dummyTrain.Values;
            hold on;
            axis([0 K+1 0 15])
            title(sprintf('Similarity: class %d Train image %d: Zebra',10,45+i));
            hold off;
        else
            subplot(2,2,i)
            bar(featureVectorTest{1+i});
            %featureVectorTest{trackTest} = dummyTest.Values;
            hold on;
            axis([0 K+1 0 15])
            title(sprintf('Similarity: class %d Test image %d: baseball',1,1+i));
            hold off;
        end
        
    end

%%TrainHistogram vs TestHistogram
%compared a set of concatTrainHistogram with TestingHistogram(1).
[concatFVTestLabel, concatFVTestDistance] = knnsearch(concatFVTest,concatFVTrain);

for i = 1:size(concatFVTestLabel,1)
    if concatFVTestLabel(i,1) >= 1 && concatFVTestLabel(i,1) < 6
        concatFVTestLabel(i) = 1;
    elseif concatFVTestLabel(i,1) >= 6 && concatFVTestLabel(i,1) < 11
        concatFVTestLabel(i) = 2;
    elseif concatFVTestLabel(i,1) >= 11 && concatFVTestLabel(i,1) < 16
        concatFVTestLabel(i) = 3;
    elseif concatFVTestLabel(i,1) >= 16 && concatFVTestLabel(i,1) < 21
        concatFVTestLabel(i) = 4;
    elseif concatFVTestLabel(i,1) >= 21 && concatFVTestLabel(i,1) < 26
        concatFVTestLabel(i) = 5;
    elseif concatFVTestLabel(i,1) >= 26 && concatFVTestLabel(i,1) < 31
        concatFVTestLabel(i) = 6;
    elseif concatFVTestLabel(i,1) >= 31 && concatFVTestLabel(i,1) < 36
        concatFVTestLabel(i) = 7;
    elseif concatFVTestLabel(i,1) >= 36 && concatFVTestLabel(i,1) < 41
        concatFVTestLabel(i) = 8;
    elseif concatFVTestLabel(i,1) >= 41 && concatFVTestLabel(i,1) < 46
        concatFVTestLabel(i) = 9;
    else
        concatFVTestLabel(i) = 10;
    end
end

%%ERROR CHECKING
errorclass = zeros(10,1);
for i = 1:size(concatFVTestLabel,1)
    if i>=1 && i <6 && concatFVTestLabel(i) ~= 1
        errorclass(1) = errorclass(1) + 1/5;
    elseif i >= 6 && i < 11 && concatFVTestLabel(i) ~= 2
        errorclass(2) = errorclass(2) + 1/5;
    elseif i >= 11 && i < 16 && concatFVTestLabel(i) ~= 3
        errorclass(3) = errorclass(3) + 1/5;
    elseif i >= 16 && i < 21 && concatFVTestLabel(i) ~= 4
        errorclass(4) = errorclass(4) + 1/5;    
    elseif i >= 21 && i < 26 && concatFVTestLabel(i) ~= 5
        errorclass(5) = errorclass(5) + 1/5;
    elseif i >= 26 && i < 31 && concatFVTestLabel(i) ~= 6
        errorclass(6) = errorclass(6) + 1/5;
    elseif i >= 31 && i < 36 && concatFVTestLabel(i) ~= 7
        errorclass(7) = errorclass(7) + 1/5;
    elseif i >= 36 && i < 41 && concatFVTestLabel(i) ~= 8
        errorclass(8) = errorclass(8) + 1/5;
    elseif i >= 41 && i < 46 && concatFVTestLabel(i) ~= 9
        errorclass(9) = errorclass(9) + 1/5;
    elseif i >= 46 && i < 51 && concatFVTestLabel(i) ~= 10
        errorclass(10) = errorclass(10) + 1/5;
    end
end

%%confusion matrix
%idk if the confusionmatrix is correct
stats = confusionmatStats(trainlabels,concatFVTestLabel);

%plotting the interest points
%showing the class butterfly(3),carplate(4),watch(9)
colors = distinguishable_colors(50);
figure(11)
for i = 1:4
    subplot(2,2,i)
    imshow(train{10+i})
    hold on;
    plot(interestTrain{10+i});
    title(sprintf('trainImage: butterfly%d with interest points',i))
    hold off;
end

figure(12)
for i = 1:4
    subplot(2,2,i)
    imshow(train{15+i})
    hold on;
    plot(interestTrain{15+i});
    title(sprintf('trainImage: carplate%d with interest points',i))
    hold off;
end

figure(13)
for i = 1:4
    subplot(2,2,i)
    imshow(train{40+i})
    hold on;
    plot(interestTrain{40+i});
    title(sprintf('trainImage: carplate%d with interest points',i))
    hold off;
end

%plot(interestTrain{1}.Location(11,1),interestTrain{1}.Location(11,2),'*');
%plot(interestTrain{1}.Location(244,1),interestTrain{1}.Location(258,2),'*')

%% For K=200 clustering
K200 = 200;
%so in here, assignments = closest centroids for each of Descriptor.
[centroid200, assignTrain200] = vl_kmeans(double(concatDesTrain), K200);

%seperate each assigned descriptors into each classes.
for i = 1:sizelabels(1)
    [closestToCentroidsTrain200{i}, distanceTrain200{i}] = knnsearch(centroid200',desTrain{i});
end

for i = 1:sizelabels(1)
    [closestToCentroidsTest200{i}, distanceTest200{i}] = knnsearch(centroid200',desTest{i});
end

concatFVTrain200 = [];
for i = 1:sizelabels(1)
    featureVectorTrain200{i} = zeros(1,K200);
    featureVectorTest200{i} = zeros(1,K200);
    %can be either size of Test or Train
    sizeDescript200 = size(closestToCentroidsTest200{i});
    for j=1:sizeDescript200(1)
        featureVectorTrain200{i}(1,closestToCentroidsTrain200{i}(j,1)) = featureVectorTrain200{i}(1,closestToCentroidsTrain200{i}(j,1)) + 1; 
        featureVectorTest200{i}(1,closestToCentroidsTest200{i}(j,1)) = featureVectorTest200{i}(1,closestToCentroidsTest200{i}(j,1)) + 1; 
    end
    % concatFVTrain's each row contains histogram of eachImages
    concatFVTrain200(i,:) = [featureVectorTrain200{i}];
    concatFVTest200(i,:) = [featureVectorTest200{i}];
end

%%TrainHistogram vs TestHistogram
%compared a set of concatTrainHistogram with TestingHistogram(1).
[concatFVTestLabel200, concatFVTestDistance200] = knnsearch(concatFVTest200,concatFVTrain200);

for i = 1:size(concatFVTestLabel200,1)
    if concatFVTestLabel200(i,1) >= 1 && concatFVTestLabel200(i,1) < 6
        concatFVTestLabel200(i) = 1;
    elseif concatFVTestLabel200(i,1) >= 6 && concatFVTestLabel200(i,1) < 11
        concatFVTestLabel200(i) = 2;
    elseif concatFVTestLabel200(i,1) >= 11 && concatFVTestLabel200(i,1) < 16
        concatFVTestLabel200(i) = 3;
    elseif concatFVTestLabel200(i,1) >= 16 && concatFVTestLabel200(i,1) < 21
        concatFVTestLabel200(i) = 4;
    elseif concatFVTestLabel200(i,1) >= 21 && concatFVTestLabel200(i,1) < 26
        concatFVTestLabel200(i) = 5;
    elseif concatFVTestLabel200(i,1) >= 26 && concatFVTestLabel200(i,1) < 31
        concatFVTestLabel200(i) = 6;
    elseif concatFVTestLabel200(i,1) >= 31 && concatFVTestLabel200(i,1) < 36
        concatFVTestLabel200(i) = 7;
    elseif concatFVTestLabel200(i,1) >= 36 && concatFVTestLabel200(i,1) < 41
        concatFVTestLabel200(i) = 8;
    elseif concatFVTestLabel200(i,1) >= 41 && concatFVTestLabel200(i,1) < 46
        concatFVTestLabel200(i) = 9;
    else
        concatFVTestLabel200(i) = 10;
    end
end

%%ERROR CHECKING
errorclass200 = zeros(10,1);
for i = 1:size(concatFVTestLabel200,1)
    if i>=1 && i <6 && concatFVTestLabel200(i) ~= 1
        errorclass200(1) = errorclass200(1) + 1/5;
    elseif i >= 6 && i < 11 && concatFVTestLabel200(i) ~= 2
        errorclass200(2) = errorclass200(2) + 1/5;
    elseif i >= 11 && i < 16 && concatFVTestLabel200(i) ~= 3
        errorclass200(3) = errorclass200(3) + 1/5;
    elseif i >= 16 && i < 21 && concatFVTestLabel200(i) ~= 4
        errorclass200(4) = errorclass200(4) + 1/5;    
    elseif i >= 21 && i < 26 && concatFVTestLabel200(i) ~= 5
        errorclass200(5) = errorclass200(5) + 1/5;
    elseif i >= 26 && i < 31 && concatFVTestLabel200(i) ~= 6
        errorclass200(6) = errorclass200(6) + 1/5;
    elseif i >= 31 && i < 36 && concatFVTestLabel200(i) ~= 7
        errorclass200(7) = errorclass200(7) + 1/5;
    elseif i >= 36 && i < 41 && concatFVTestLabel200(i) ~= 8
        errorclass200(8) = errorclass200(8) + 1/5;
    elseif i >= 41 && i < 46 && concatFVTestLabel200(i) ~= 9
        errorclass200(9) = errorclass200(9) + 1/5;
    elseif i >= 46 && i < 51 && concatFVTestLabel200(i) ~= 10
        errorclass200(10) = errorclass200(10) + 1/5;
    end
end
%confusion matrix K=200
%idk if the confusionmatrix is correct
stats200 = confusionmatStats(trainlabels,concatFVTestLabel200);

%% what to show
%For K=300
fprintf('-----------------------------------------------\r')
fprintf('%s %d\r','K =',K)
fprintf('%s %d\r','Number of Train Images for each class =',5)
fprintf('%s %d\r','Number of Testing Images for each class =',5)
fprintf('-----------------------------------------------\r')
fprintf('%s %.1f%s\n','According to confusionmatStats, Average accuracy of reconizing is: ',mean(stats.accuracy)*100, '%.');
fprintf('%s\n','According to computational error I computed:');
fprintf('\t%s%.1f\t%s%.1f\r','Class1: ',errorclass(1),'Class2: ',errorclass(2));
fprintf('\t%s%.1f\t%s%.1f\r','Class3: ',errorclass(3),'Class4: ',errorclass(4));
fprintf('\t%s%.1f\t%s%.1f\r','Class5: ',errorclass(5),'Class6: ',errorclass(6));
fprintf('\t%s%.1f\t%s%.1f\r','Class7: ',errorclass(7),'Class8: ',errorclass(8));
fprintf('\t%s%.1f\t%s%.1f\r','Class9: ',errorclass(9),'Class10: ',errorclass(10));
fprintf('\r%s','ConfusionMatrix is:')
stats.confusionMat

fprintf('-----------------------------------------------\r')
fprintf('%s %d\r','K =',K200)
fprintf('%s %d\r','Number of Train Images for each class =',5)
fprintf('%s %d\r','Number of Testing Images for each class =',5)
fprintf('-----------------------------------------------\r')
fprintf('%s %.1f%s\n','According to confusionmatStats, Average accuracy of reconizing is: ',mean(stats.accuracy)*100, '%.');
fprintf('%s\n','According to computational error I computed:');
fprintf('\t%s%.1f\t%s%.1f\r','Class1: ',errorclass200(1),'Class2: ',errorclass200(2));
fprintf('\t%s%.1f\t%s%.1f\r','Class3: ',errorclass200(3),'Class4: ',errorclass200(4));
fprintf('\t%s%.1f\t%s%.1f\r','Class5: ',errorclass200(5),'Class6: ',errorclass200(6));
fprintf('\t%s%.1f\t%s%.1f\r','Class7: ',errorclass200(7),'Class8: ',errorclass200(8));
fprintf('\t%s%.1f\t%s%.1f\r','Class9: ',errorclass200(9),'Class10: ',errorclass200(10));
fprintf('\r%s','ConfusionMatrix is:')
stats200.confusionMat


