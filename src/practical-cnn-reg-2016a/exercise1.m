setup() ;

%% Part 1.1: convolution

%% Part 1.1.1: convolution by a single filter

% Load an image and convert it to gray scale and single precision
x = im2single(rgb2gray(imread('data/ray.jpg'))) ;

%x = single([3 1 4 3; 2 3 2 1 ; 1 2 5 6; 1 9 5 3; 2 5 3 6;1 2 4 4; 5 2 3 5])';% 3 0 8 4 2]);
%x = rand(7,70);

% Define a filter
w = single([
   0 -1  0 
  -1  4 -1  
   0 -1  0 ]) ;

%Apply the filter to the image
y = vl_nnconv(x, w, []) ;
%%
% Visualize the results
figure(12) ; clf ; colormap gray ;
set(gcf, 'name', 'Part 1.1: convolution') ;

subplot(2,2,1) ;
imagesc(x) ;
axis off image ;
title('Input image x') ;

y1 = vl_nnconv(x,w2,[]);
subplot(2,2,2) ;
imagesc(w2) ;
axis off image ;
title('Filter w') ;

subplot(2,2,3) ;
imagesc(y1) ;
axis off image ;
title('Output image y') ;

%% Part 1.1.2: convolution by a bank of filters

% Concatenate three filters in a bank
w1 = single([
   0 -1  0
  -1  4 -1
   0 -1  0]) ;

w2 = single([
  -1 0 +1
  -1 0 +1
  -1 0 +1]) ;

w3 = single([
  -1 -1 -1
   0  0  0
  +1 +1 +1]) ;

%4-D wbank
wbank = cat(5, w1, w2, w3) ;

% Apply convolution
y = vl_nnconv(x, wbank, []);
%%
% Show feature channels
figure(12) ; clf('reset') ;
set(gcf, 'name', 'Part 1.1.2: channels') ;
colormap gray ;
showFeatureChannels(x) ;

%% Part 1.1.3: convolving a batch of images

x1 = im2single(rgb2gray(imread('data/ray.jpg'))) ;
x2 = im2single(rgb2gray(imread('data/crab.jpg'))) ;
x = cat(4, x1, x2) ;

y = vl_nnconv(x, wbank, []) ;

figure(13) ; clf('reset') ; colormap gray ;
set(gcf, 'name', 'Part 1.1.3: filtering a batch') ;

subplot(4,2,1) ; imagesc(x1) ; axis off image ;
subplot(4,2,3) ; imagesc(y(:,:,1,1)) ; axis off image ;
subplot(4,2,5) ; imagesc(y(:,:,2,1)) ; axis off image ;
subplot(4,2,7) ; imagesc(y(:,:,3,1)) ; axis off image ;

subplot(4,2,2) ; imagesc(x2) ; axis off image ;
subplot(4,2,4) ; imagesc(y(:,:,1,2)) ; axis off image ;
subplot(4,2,6) ; imagesc(y(:,:,2,2)) ; axis off image ;
subplot(4,2,8) ; imagesc(y(:,:,3,2)) ; axis off image ;

%% Part 1.2: non-linear activation functions (ReLU)

%% Part 1.2.1: Laplacian and ReLU
x = im2single(rgb2gray(imread('data/ray.jpg'))) ;

% Convolve with the negated Laplacian
y = vl_nnconv(x,  -w, []) ;

% Apply the ReLU operator
z = vl_nnrelu(y) ;

figure(14) ; clf ; set(gcf, 'name', 'Part 1.2.1: Laplacian and ReLU') ;
colormap gray ;
subplot(2,2,1); imagesc(x) ; axis off image ; title('Image x') ;
subplot(2,2,2); imagesc(y) ; axis off image ; title('Laplacian y')
subplot(2,2,3); imagesc(z) ; axis off image ; title('ReLU z') ;

%% Part 1.2.2: effect of adding a bias

bias = single(-0.2) ;
y = vl_nnconv(x, - w, bias) ;
y1 = vl_nnconv(x, -w, single(-0.1));
z = vl_nnrelu(y) ;

figure(17) ; clf ; set(gcf, 'name', 'Part 1.2.2: adding a bias') ;
colormap gray ;
subplot(2,2,1); imagesc(x) ; axis off image ; title('Image x') ;
subplot(2,2,2); imagesc(y) ; axis off image ; title('Laplacian y with bias')
subplot(2,2,3); imagesc(z) ; axis off image ; title('ReLU z') ;
