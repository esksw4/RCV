%% 2: PCA recognition
%eigs

clear;
close all;
clc;

imgDir = '/Users/e.kim4/Documents/MATLAB/RCV_project4/vision_dataset2';
imds = imageDatastore(imgDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[testFace,trainFace] = splitEachLabel(imds,5);
trainlabels = grp2idx(trainFace.Labels);
sizelabels = size(trainlabels);

for i = 1:sizelabels(1)
    % each img is N=243 x P=320 M = numberofimages = 10 per class
    % 100 for whole classes
    trainF{i} = imread(trainFace.Files{i});
    testF{i} = imread(testFace.Files{i});
    if size(trainF{i},3)== 3
        trainF{i} = rgb2gray(imread(trainFace.Files{i}));
    elseif size(testF{i},3) ==3
        testF{i} = rgb2gray(imread(testFace.Files{i}));
    end

    if i == 1
        % has (N*P)x M dimension
        trainA = zeros(prod(size(trainF{1})),sizelabels(1));
        testA = zeros(prod(size(testF{1})),sizelabels(1));
        trainMean = zeros(size(trainF{1}));
        %testMean = zeros(size(testF{1}));
    end
    %avgface has NxP dimension
    trainMean = trainMean + double(trainF{i});
    %testMean = testMean + double(trainF{i});
    trainA(:,i) = trainF{i}(:);
    testA(:,i) = testF{i}(:);
end;
%%compute averageface to show; averageface for computation
trainMean = mean(trainA,2);
%testMean = mean(testA,2);

%%compute the differences of original image - Mean face
%removing all common face features that the faces share together 
%so that each face is left with each unique features
for i = 1:size(trainA,2)
    %(N*P) x M
    %subtract each column with averageface
    trainA_Mean(:,i) = trainA(:,i) - trainMean;
    testA_Mean(:,i) = testA(:,i) - trainMean;
end

%%compute covariance Matrix & get eigenvector and eigenvalues
%originalC = A_Mean * transpose(A_Mean);
%better to reduce the dimensionality to reduce noise and
%number of computation
trainReducedC = transpose(trainA_Mean)*trainA_Mean;
testReducedC = transpose(testA_Mean)*testA_Mean;

%%Choosing K using SVD
%such that K<=M and represent whole training set
%columns of U = eigenvectors
%S = eigenvalues
[trainU,trainS,trainV] = svd(trainReducedC);
[testU,testS,testV] = svd(testReducedC);
figure(1)
subplot(1,2,1)
plot(trainS);
subplot(1,2,2)
plot(testS);
hold off;
title('Decay of Eigenvalues')
xlabel('NM')
ylabel('eigenvalues')
%%
%find rank R = K
trainK = rank(trainReducedC);
testK = rank(testReducedC);

%%get eigenvector
trainEVecReduced = trainU(:,1:trainK);
testEVecReduced = testU(:,1:testK);

%%convert reduced dimensional K eigenvectors to origianl dimensionality
%u_i = A*v_i;
trainEvecOriginal(:,1:trainK) = trainA_Mean*trainEVecReduced(:,1:trainK);
testEvecOriginal(:,1:testK) = testA_Mean*testEVecReduced(:,1:testK);

%%normalized each column
%trainEvecOriginal=normc(trainEvecOriginal);
%testEvecOriginal=normc(testEvecOriginal);

%%finding coefficient a by dotproduct
trainCoef = (trainA_Mean'*trainEvecOriginal);
testCoef = (testA_Mean'*testEvecOriginal);

%normalized each column
trainCoef = normc(trainCoef);
testCoef = normc(testCoef);

%%reconstruc with eigenfaces.
%originalFace = avgfaceForComp + a(1)*eigvecOriginal(:,1)'+...+a(114)*eigvecOriginal(114);
testFace = []
for i=1:trainK
    if i == 1
        testFace = repmat(trainMean',sizelabels(1),1) + testCoef(:,i)*testEvecOriginal(:,i)';
    end
    testFace = testFace + testCoef(:,i)*testEvecOriginal(:,i)';
end
%%
figure(1)
subplot(5,2,1)
imshow(uint8(reshape(testFace(1,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 1:',testK))
subplot(5,2,2)
imshow(uint8(reshape(testFace(6,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 6:',testK))
subplot(5,2,3)
imshow(uint8(reshape(testFace(11,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 11:',testK))
subplot(5,2,4)
imshow(uint8(reshape(testFace(16,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 16:',testK))
subplot(5,2,5)
imshow(uint8(reshape(testFace(21,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 21:',testK))
subplot(5,2,6)
imshow(uint8(reshape(testFace(26,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 26:',testK))
subplot(5,2,7)
imshow(uint8(reshape(testFace(31,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 31:',testK))
subplot(5,2,8)
imshow(uint8(reshape(testFace(36,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 36:',testK))
subplot(5,2,9)
imshow(uint8(reshape(testFace(41,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 41:',testK))
subplot(5,2,10)
imshow(uint8(reshape(testFace(46,:),[size(testF{1})])));
title(sprintf('K = %d: trainImage 46:',testK))

