function HandGestureRecognition

str1='Y';
str='Y';

url = 'E:\Final Project_v2\ImageProcessing-master';
outputFolder = 'E:\Final Project_v2\ImageProcessing-master';

rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'ones', 'twos', 'threes', 'fours' , 'fives' };

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)


minSetCount = min(tbl{:,2}); 


imds = splitEachLabel(imds, minSetCount, 'randomize');


countEachLabel(imds)



cnnMatFile = 'E:\Final Project_v2\imagenet-caffe-alex.mat';
convnet = helperImportMatConvNet(cnnMatFile);


%convnet.Layers
	% 23x1 Layer array with layers:

     %1   'input'                 Image Input                   227x227x3 images with 'zerocenter' normalization
     %2   'conv1'                 Convolution                   96 11x11x3 convolutions with stride [4  4] and padding [0  0]
     %3   'relu1'                 ReLU                          ReLU
     %4   'norm1'                 Cross Channel Normalization   cross channel normalization with 5 channels per element
     %5   'pool1'                 Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0]
     %6   'conv2'                 Convolution                   256 5x5x48 convolutions with stride [1  1] and padding [2  2]
     %7   'relu2'                 ReLU                          ReLU
     %8   'norm2'                 Cross Channel Normalization   cross channel normalization with 5 channels per element
     %9   'pool2'                 Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0]
    %10   'conv3'                 Convolution                   384 3x3x256 convolutions with stride [1  1] and padding [1  1]
    %11   'relu3'                 ReLU                          ReLU
    %12   'conv4'                 Convolution                   384 3x3x192 convolutions with stride [1  1] and padding [1  1]
    %13   'relu4'                 ReLU                          ReLU
    %14   'conv5'                 Convolution                   256 3x3x192 convolutions with stride [1  1] and padding [1  1]
    %15   'relu5'                 ReLU                          ReLU
    %16   'pool5'                 Max Pooling                   3x3 max pooling with stride [2  2] and padding [0  0]
    %17   'fc6'                   Fully Connected               4096 fully connected layer
    %18   'relu6'                 ReLU                          ReLU
    %19   'fc7'                   Fully Connected               4096 fully connected layer
    %20   'relu7'                 ReLU                          ReLU
    %21   'fc8'                   Fully Connected               1000 fully connected layer
    %22   'prob'                  Softmax                       softmax
    %23   'classificationLayer'   Classification Output         cross-entropy with 'n01440764', 'n01443537', and 998 other classes





% Pre-process Images For CNN
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);


function Iout = readAndPreprocessImage(filename)

        I = imread(filename);

        if ismatrix(I)  %If image is grayscale
            I = cat(3,I,I,I);
        end

        
        Iout = imresize(I, [227 227]);

       
    end
	
	
	
	
	[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');

	

w1 = convnet.Layers(2).Weights;
w1 = mat2gray(w1);
w1 = imresize(w1,5);

featureLayer = 'fc7';
%Extract Training Features
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
	
	
trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


%Extract Test Features
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);


predictedLabels = predict(classifier, testFeatures);


testLabels = testSet.Labels;


confMat = confusionmat(testLabels, predictedLabels);


confMat = bsxfun(@rdivide,confMat,sum(confMat,2))


mean(diag(confMat))
disp('Training Done');

while(strcmp(str,str1))
capture
newImage = fullfile(rootFolder, 'test', 'test1.jpg');

img = readAndPreprocessImage(newImage);

imageFeatures = activations(convnet, img, featureLayer);

label = predict(classifier, imageFeatures)
pause(4);
perform
pause(10);
disp('Gesture 1 for new input and Gesture 3 to end');
capture
newImage = fullfile(rootFolder, 'test', 'test1.jpg');

img = readAndPreprocessImage(newImage);

imageFeatures = activations(convnet, img, featureLayer);

label = predict(classifier, imageFeatures)
pause(4);
if(label=='ones')
disp('Get Ready For New Input');
pause(2);
str='Y';
end
if(label=='threes')
disp('Exiting');
str='N';
end
if isempty(str)
    str = 'N';
end

end

function capture
vid=videoinput('winvideo',1);                                             %sets videoinput to the webcam, and the webcam device 1
set(vid,'ReturnedColorspace','rgb')
pause(2);                                                                 % pause 2 seconds before snapshot of background image
IM1=getsnapshot(vid);                                                     %get snapshot from the webcam video and store to IM1 variable

subplot(1,1,1);
imshow(IM1);title('Gesture');                 %open up a figure and show the image stored in IM1 variable
pause(3);
close all;
imwrite(IM1,'E:\Final Project_v2\ImageProcessing-master\101_ObjectCategories\test\test1.jpg');
end

function perform
global key;
import java.awt.Robot;
import java.awt.event.*;
key=Robot;


if(label=='ones')
%ZOOM IN
disp('ZOOM_IN');
key.keyPress  (java.awt.event.KeyEvent.VK_ALT );
            key.keyPress  (java.awt.event.KeyEvent.VK_ESCAPE );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_ALT );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ESCAPE );
            pause(2);
           
            key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_ADD);
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ADD);
 key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_ADD);
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ADD);
             key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_ADD);
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ADD);
end
if(label=='twos')
    disp('ZOOM OUT');
key.keyPress  (java.awt.event.KeyEvent.VK_ALT );
            key.keyPress  (java.awt.event.KeyEvent.VK_ESCAPE );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_ALT );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ESCAPE );
            pause(2);
            
            key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_SUBTRACT );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_SUBTRACT );
 key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_SUBTRACT );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_SUBTRACT );
             key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_SUBTRACT );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_SUBTRACT );

end
if(label=='threes')
    disp('PREVIOUS');
key.keyPress  (java.awt.event.KeyEvent.VK_ALT );
            key.keyPress  (java.awt.event.KeyEvent.VK_ESCAPE );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_ALT );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ESCAPE );
            pause(2);
            key.keyPress  (java.awt.event.KeyEvent.VK_LEFT);
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_LEFT );



end
if(label=='fours')
    disp('NEXT');
key.keyPress  (java.awt.event.KeyEvent.VK_ALT );
            key.keyPress  (java.awt.event.KeyEvent.VK_ESCAPE );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_ALT );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ESCAPE );
            pause(2);
            
              key.keyPress  (java.awt.event.KeyEvent.VK_RIGHT);
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_RIGHT );

end

if(label=='fives')
    disp('ROTATE');
key.keyPress  (java.awt.event.KeyEvent.VK_ALT );
            key.keyPress  (java.awt.event.KeyEvent.VK_ESCAPE );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_ALT );
            key.keyRelease  (java.awt.event.KeyEvent.VK_ESCAPE );
            pause(2);
            
			 key.keyPress  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyPress  (java.awt.event.KeyEvent.VK_COMMA );
      
             key.keyRelease  (java.awt.event.KeyEvent.VK_CONTROL );
            key.keyRelease  (java.awt.event.KeyEvent.VK_COMMA );
              
end




end
end
