%转换cell tracking的groundtruth：二值图——>x.txt(class x_center y_center width heigh)

framesPath = 'F:\wyw\课程\计算机视觉\课程项目作业\细胞跟踪\DIC-C2DH-HeLa (train)\02_ST\SEG\';%图像序列所在路径，同时要保证图像大小相同  

imageDirPath =  framesPath;  %存储图像的文件夹路径
fileExt = '*.tif';  %待读取图像的后缀名
%获取所有路径
files = dir(fullfile(imageDirPath,fileExt)); 
num = size(files,1);

%读入图片  
startFrame = 0; %从哪一帧开始  
endFrame =size(files,1)-1; %哪一帧结束  

for idx=startFrame:endFrame
    
    fileName=sprintf('man_seg%.3d',idx);
    fileName_out_txt=sprintf('%.6d',idx);
    img=imread([framesPath,fileName,'.tif']);  
%     figure;imshow(img);title('origin');
    [m,n]=size(img);
    img_size=size(img);
    for i=1:m
        for j=1:n
            if img(i,j)>0
                imgnew(i,j)=255;
            else
                imgnew(i,j)=0;
            end
        end
    end
%     figure;imshow(imgnew);title('groundtruth_01');
%% 获取图像中的连通域，及其中心点

    %用ostu方法获取二值化阈值，进行二值化并进行显示(将512x512 double转成512x512 logical)
    level=graythresh(imgnew);
    openbw=im2bw(imgnew,level);

    %获取区域的'basic'属性， 'Area', 'Centroid', and 'BoundingBox' 
    stats = regionprops(openbw, 'BoundingBox' ,'Area','Centroid' ,'PixelList','PixelIdxList'); %统计白色的连通区域
    centroids = cat(1, stats.Centroid);  % 所有连通域的重心点的坐标
    boundingboxes=cat(1, stats.BoundingBox);
   
    fid = fopen([framesPath,fileName_out_txt,'.txt'], 'wt'); 
    for i=1:size(stats)
         centroid=[centroids(i,1), centroids(i,2)];               %每个连通区域的重心位置
         normalized=normalization(img_size, centroid);
        
         fprintf(fid,'0 %f %f %f %f\n',normalized); 
    end
    fclose(fid);  
end  

function normalized=normalization_boundingbox(img_size, boundingbox)
    %得到长和宽的缩放比
    dw = 1./(img_size(1));
    dh = 1./(img_size(2));
    
    %分别计算中心点坐标，框的宽和高
    %boundingbox=[The_upper_left_corner_x,The_upper_left_corner_y,x_width,y_width]
    x = boundingbox(1) + boundingbox(3)/2.0 ;
    y = boundingbox(2) + boundingbox(4)/2.0 ;
    w = boundingbox(3);
    h = boundingbox(4);
        
    %按照图片长和宽进行归一化
    x = x * dw;
    w = w * dw;
    y = y * dh;
    h = h * dh;
    normalized=[x,y,w,h];
end


function normalized=normalization(img_size, centroid)
    %得到长和宽的缩放比
    dw = 1./(img_size(1));
    dh = 1./(img_size(2));
    
    %分别计算中心点坐标，框的宽和高
    x = centroid(1);
    y = centroid(2);
    w = 150;
    h = 150;
        
    %按照图片长和宽进行归一化
    x = x * dw;
    w = w * dw;
    y = y * dh;
    h = h * dh;
    normalized=[x,y,w,h];
end
