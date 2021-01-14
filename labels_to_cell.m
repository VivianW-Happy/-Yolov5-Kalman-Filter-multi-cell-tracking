%ת��cell tracking��groundtruth����ֵͼ����>x.txt(class x_center y_center width heigh)

framesPath = 'F:\wyw\�γ�\������Ӿ�\�γ���Ŀ��ҵ\ϸ������\DIC-C2DH-HeLa (train)\02_ST\SEG\';%ͼ����������·����ͬʱҪ��֤ͼ���С��ͬ  

imageDirPath =  framesPath;  %�洢ͼ����ļ���·��
fileExt = '*.tif';  %����ȡͼ��ĺ�׺��
%��ȡ����·��
files = dir(fullfile(imageDirPath,fileExt)); 
num = size(files,1);

%����ͼƬ  
startFrame = 0; %����һ֡��ʼ  
endFrame =size(files,1)-1; %��һ֡����  

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
%% ��ȡͼ���е���ͨ�򣬼������ĵ�

    %��ostu������ȡ��ֵ����ֵ�����ж�ֵ����������ʾ(��512x512 doubleת��512x512 logical)
    level=graythresh(imgnew);
    openbw=im2bw(imgnew,level);

    %��ȡ�����'basic'���ԣ� 'Area', 'Centroid', and 'BoundingBox' 
    stats = regionprops(openbw, 'BoundingBox' ,'Area','Centroid' ,'PixelList','PixelIdxList'); %ͳ�ư�ɫ����ͨ����
    centroids = cat(1, stats.Centroid);  % ������ͨ������ĵ������
    boundingboxes=cat(1, stats.BoundingBox);
   
    fid = fopen([framesPath,fileName_out_txt,'.txt'], 'wt'); 
    for i=1:size(stats)
         centroid=[centroids(i,1), centroids(i,2)];               %ÿ����ͨ���������λ��
         normalized=normalization(img_size, centroid);
        
         fprintf(fid,'0 %f %f %f %f\n',normalized); 
    end
    fclose(fid);  
end  

function normalized=normalization_boundingbox(img_size, boundingbox)
    %�õ����Ϳ�����ű�
    dw = 1./(img_size(1));
    dh = 1./(img_size(2));
    
    %�ֱ�������ĵ����꣬��Ŀ�͸�
    %boundingbox=[The_upper_left_corner_x,The_upper_left_corner_y,x_width,y_width]
    x = boundingbox(1) + boundingbox(3)/2.0 ;
    y = boundingbox(2) + boundingbox(4)/2.0 ;
    w = boundingbox(3);
    h = boundingbox(4);
        
    %����ͼƬ���Ϳ���й�һ��
    x = x * dw;
    w = w * dw;
    y = y * dh;
    h = h * dh;
    normalized=[x,y,w,h];
end


function normalized=normalization(img_size, centroid)
    %�õ����Ϳ�����ű�
    dw = 1./(img_size(1));
    dh = 1./(img_size(2));
    
    %�ֱ�������ĵ����꣬��Ŀ�͸�
    x = centroid(1);
    y = centroid(2);
    w = 150;
    h = 150;
        
    %����ͼƬ���Ϳ���й�һ��
    x = x * dw;
    w = w * dw;
    y = y * dh;
    h = h * dh;
    normalized=[x,y,w,h];
end
