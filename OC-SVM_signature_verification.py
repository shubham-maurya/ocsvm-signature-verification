
# coding: utf-8

# In[1]:

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn import preprocessing
from sklearn import svm


# ## Original Image

# In[2]:

img = cv2.imread('D:\\MY-DOC\\Desktop\\signatures\\full_org\\original_1_1.png',0)
#img = cv2.imread('D:\\MY-DOC\\Desktop\\testing.png',0)
plt.imshow(img,'gray')
plt.show()


# ## Applying Gaussian Blurring and Otsu Thresholding for Binarization

# In[3]:

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image=np.invert(th3)
plt.imshow(image,'gray')
plt.show()


# ## Cropping image to get bounding box

# In[4]:

def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

cimg=crop_image(image,tol=0)
plt.imshow(cimg,'gray')
plt.show()


# ## Thinning the image

# In[5]:

def thinning(img):
    #img = cimg
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

  #  plt.imshow(skel,'gray')
   # plt.show()
    return skel

timg=thinning(cimg)
plt.imshow(timg,'gray')
plt.show()


# ## Finding number of white pixels and connected components in signature

# In[6]:

#Feature generation area
area=cv2.countNonZero(cimg)
print(area)
img1=np.invert(cimg)
cc=cv2.connectedComponents(img1)[0]
print(cc)


# ## Dividing image into 4 parts

# In[7]:

def coords(timg):
    rows,cols=timg.shape
    img_tl=timg[0:int(rows/2),0:int(cols/2)]
    img_tr=timg[int(rows/2)+1:rows,0:int(cols/2)]
    img_bl=timg[0:int(rows/2),int(cols/2)+1:cols]
    img_br=timg[int(rows/2)+1:rows,int(cols/2)+1:cols]
    
    tl_x,tl_y=COG(img_tl)
    tr_x,tr_y=COG(img_tr)
    bl_x,bl_y=COG(img_bl)
    br_x,br_y=COG(img_br)

    return tl_x,tl_y,tr_x,tr_y,bl_x,bl_y,br_x,br_y



rows,cols=timg.shape
img_tl=timg[0:int(rows/2),0:int(cols/2)]
img_tr=timg[int(rows/2)+1:rows,0:int(cols/2)]
img_bl=timg[0:int(rows/2),int(cols/2)+1:cols]
img_br=timg[int(rows/2)+1:rows,int(cols/2)+1:cols]

#fig,ax=plt.subplots(nrows=1,ncols=1)
#plt.subplot(2,2,1)    
plt.imshow(img_tl,'gray')
plt.show()

plt.imshow(img_tr,'gray')
plt.show()

plt.imshow(img_bl,'gray')
plt.show()

plt.imshow(img_br,'gray')
plt.show()

  




# In[ ]:




# ## Finding Centre of gravity of each sub-image

# In[8]:

def COG(img):
    x_cor=0
    xrun_sum=0
    y_cor=0
    yrun_sum=0
    #print(img.shape)
    for i in range(img.shape[0]):
        x_cor+=sum(img[i])*i/255
        xrun_sum+=sum(img[i])/255

    for i in range(img.shape[1]):
        y_cor+=sum(img[:,i])*i/255
        yrun_sum+=sum(img[:,i])/255
        #print(img.shape[1]) 
        if yrun_sum==0:
            x_pos=0
        else:
            x_pos=y_cor/(yrun_sum)
        if xrun_sum==0:
            y_pos=0
        else:
            y_pos=x_cor/(xrun_sum)
        
   # print(x_pos)
  #  print(y_pos)
    
    return (x_pos/img.shape[1],y_pos/img.shape[0])
COG(img_bl)


# ## Generating feature dataset of original images!

# In[54]:

data=[]
#df=pd.DataFrame(columns=["Writer_no","Sample_no","area","connected_comps"])
for i in range(1,56):
   # print(i)
    for j in range(1,25):
        path=f'D:\\MY-DOC\\Desktop\\signatures\\full_org\\original_{i}_{j}.png'
        img = cv2.imread(path,0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image=np.invert(th3)
        
        cimg=crop_image(image,tol=0)
        area=cv2.countNonZero(cimg)/(cimg.shape[0]*cimg.shape[1])
        #Find proportion of white cells
        img1=np.invert(cimg)
        
        cc=cv2.connectedComponents(img1)[0]
        #Generate connected components
        
        timg=thinning(cimg)
        #Thinning the image!
        
        tl_x,tl_y,tr_x,tr_y,bl_x,bl_y,br_x,br_y=coords(timg)
        #Extracting features
        
        x=pd.Series([i,j,area,cc,tl_x,tl_y,tr_x,tr_y,bl_x,bl_y,br_x,br_y],index=
                    ["Writer_no","Sample_no","area","connected_comps","tl_x","tl_y","tr_x","tr_y","bl_x","bl_y",
                    "br_x","br_y"])
        data.append(x)
df=pd.DataFrame(data)


# ## Original feature dataset

# In[65]:




# In[55]:

df.head()


# ## Generating forgery feature dataset

# In[28]:

data_f=[]
#df=pd.DataFrame(columns=["Writer_no","Sample_no","area","connected_comps"])
for i in range(1,56):
   # print(i)
    for j in range(1,25):
        path=f'D:\\MY-DOC\\Desktop\\signatures\\full_forg\\forgeries_{i}_{j}.png'
        img = cv2.imread(path,0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image=np.invert(th3)
        
        cimg=crop_image(image,tol=0)
        area=cv2.countNonZero(cimg)/(cimg.shape[0]*cimg.shape[1])
        #Find proportion of white cells
        img1=np.invert(cimg)
        
        cc=cv2.connectedComponents(img1)[0]
        #Generate connected components
        
        timg=thinning(cimg)
        #Thinning the image!
        
        tl_x,tl_y,tr_x,tr_y,bl_x,bl_y,br_x,br_y=coords(timg)
        #Extracting features
        
        x=pd.Series([i,j,area,cc,tl_x,tl_y,tr_x,tr_y,bl_x,bl_y,br_x,br_y],index=
                    ["Writer_no","Sample_no","area","connected_comps","tl_x","tl_y","tr_x","tr_y","bl_x","bl_y",
                    "br_x","br_y"])
        data_f.append(x)
df_f=pd.DataFrame(data_f)


# ## Forgery features dataset

# In[29]:

df_f.head()


# ## Generating alternate feature dataset
# ### Splitting the image into 8 parts, then calculating tan inverse of centre of gravity of each part

# In[9]:

def tan_i(x):
    #print(x)
    if x[0]==0:
        return 90
    return math.degrees(math.atan(x[1]/x[0]))

def alt_coords(timg):
    rows,cols=timg.shape
    
    img_tl1=timg[0:int(rows/2),0:int(cols/4)]
    img_tl2=timg[0:int(rows/2),int(cols/4)+1:int(cols/2)]
    
    img_tr1=timg[0:int(rows/2),int(cols/2)+1:int(0.75*cols)]
    img_tr2=timg[0:int(rows/2),int(0.75*cols)+1:cols]
    
    img_bl1=timg[int(rows/2)+1:rows,0:int(cols/4)]
    img_bl2=timg[int(rows/2)+1:rows,int(cols/4)+1:int(cols/2)]
    
    img_br1=timg[int(rows/2)+1:rows,int(cols/2)+1:int(0.75*cols)]
    img_br2=timg[int(rows/2)+1:rows,int(0.75*cols)+1:cols]
    

    #plt.imshow(timg,'gray')
    #plt.show()
    
    tl1=tan_i(COG(img_tl1))
    tl2=tan_i(COG(img_tl2))
    tr1=tan_i(COG(img_tr1))
    tr2=tan_i(COG(img_tr2))
    bl1=tan_i(COG(img_bl1))
    bl2=tan_i(COG(img_bl2))
    br1=tan_i(COG(img_br1))
    br2=tan_i(COG(img_br2))
    
    #plt.imshow(img_br1,'gray')
    #plt.show()
    #print(COG(img_br1))
    return tl1,tl2,tr1,tr2,bl1,bl2,br1,br2

alt_coords(timg)


# ## Generating alternate original feature dataset

# In[10]:

alt_data=[]
#df=pd.DataFrame(columns=["Writer_no","Sample_no","area","connected_comps"])
for i in range(1,56):
   # print(i)
    for j in range(1,25):
        path=f'D:\\MY-DOC\\Desktop\\signatures\\full_org\\original_{i}_{j}.png'
        img = cv2.imread(path,0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image=np.invert(th3)
        
        cimg=crop_image(image,tol=0)
        area=cv2.countNonZero(cimg)/(cimg.shape[0]*cimg.shape[1])
        #Find proportion of white cells
        img1=np.invert(cimg)
        
        cc=cv2.connectedComponents(img1)[0]
        #Generate connected components
        
        timg=thinning(cimg)
        #Thinning the image!
        
        tl1,tl2,tr1,tr2,bl1,bl2,br1,br2=alt_coords(timg)
        #Extracting features
        
        x=pd.Series([i,j,area,cc,tl1,tl2,tr1,tr2,bl1,bl2,br1,br2],index=
                    ["Writer_no","Sample_no","area","connected_comps","tl1","tl2","tr1","tr2","bl1","bl2",
                    "br1","br2"])
        alt_data.append(x)
alt_df=pd.DataFrame(alt_data)


# ## Alternate original dataset

# In[11]:

alt_df.head()


# ## Alternate forgery feature dataset generation

# In[12]:

alt_data_f=[]
#df=pd.DataFrame(columns=["Writer_no","Sample_no","area","connected_comps"])
for i in range(1,56):
   # print(i)
    for j in range(1,25):
        path=f'D:\\MY-DOC\\Desktop\\signatures\\full_forg\\forgeries_{i}_{j}.png'
        img = cv2.imread(path,0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image=np.invert(th3)
        
        cimg=crop_image(image,tol=0)
        area=cv2.countNonZero(cimg)/(cimg.shape[0]*cimg.shape[1])
        #Find proportion of white cells
        img1=np.invert(cimg)
        
        cc=cv2.connectedComponents(img1)[0]
        #Generate connected components
        
        timg=thinning(cimg)
        #Thinning the image!
        
        tl1,tl2,tr1,tr2,bl1,bl2,br1,br2=alt_coords(timg)
        #Extracting features
        
        x=pd.Series([i,j,area,cc,tl1,tl2,tr1,tr2,bl1,bl2,br1,br2],index=
                    ["Writer_no","Sample_no","area","connected_comps","tl1","tl2","tr1","tr2","bl1","bl2",
                    "br1","br2"])
        alt_data_f.append(x)
alt_df_f=pd.DataFrame(alt_data_f)


# ## Alternate forgery dataset

# In[13]:

alt_df_f.head()


# # Using One-Class SVM

# 

# In[ ]:




# ## Pre-processing data

# In[330]:


alt_df.area=preprocessing.scale(alt_df.area)
alt_df.connected_comps=preprocessing.scale(alt_df.connected_comps)
alt_df.tl1=preprocessing.scale(alt_df.tl1)
alt_df.tl2=preprocessing.scale(alt_df.tl2)
alt_df.tr1=preprocessing.scale(alt_df.tr1)
alt_df.tr2=preprocessing.scale(alt_df.tr2)
alt_df.bl1=preprocessing.scale(alt_df.bl1)
alt_df.bl2=preprocessing.scale(alt_df.bl2)
alt_df.br1=preprocessing.scale(alt_df.br1)
alt_df.br2=preprocessing.scale(alt_df.br2)


# ## Choosing dataset

# In[331]:

os=24*32
data=alt_df.iloc[0+os:14+os]
print(data.shape)
data=data.drop("Writer_no",axis=1)
data=data.drop("Sample_no",axis=1)


# ## Tuning SVM for optimal parameters

# In[352]:

def tuning_SVM(data):
    max1=0
    maxsv=0
    best_nu=0
    best_gamma=0
    for i in range(1,100):
        for j in range(1,100):
            clf = svm.OneClassSVM(nu=0.01*i, kernel="rbf", gamma=0.005*j)
            clf.fit(data)
            x=clf.predict(data)
            #print(clf.decision_function(data))
            if (len(x[x==1]) > max1) and len(clf.support_vectors_) > maxsv :
                best_nu=0.01*i
                best_gamma=0.005*j
                #print(len(x[x==1]))
                #print(len(clf.support_vectors_))
                max1=len(x[x==1])
                maxsv=len(clf.support_vectors_)

    return best_nu,best_gamma
#print("Optimal parameters are ",best_nu,best_gamma)


# ## Generating test dataset

# In[333]:

li=[os+i for i in range(0,24)]
test_m=alt_df.drop(alt_df.index[li])
rand_signs=test_m.sample(n=20)


# In[334]:

test=alt_df.iloc[14+os:24+os]
test=test.append(rand_signs)
#print(test)
#test=pd.DataFrame(test.append(alt_df.iloc[888]))
#for i in range(1,21):
#    test=test.append(alt_df.iloc[i*30])

test=test.drop("Writer_no",axis=1)
test=test.drop("Sample_no",axis=1)

print(test.shape)


# In[376]:

clf = svm.OneClassSVM(nu=best_nu, kernel="rbf", gamma=best_gamma)
clf.fit(data)
preds=clf.predict(test)
print(preds)

pdf1=clf.decision_function(test[0:10])
pdf2=clf.decision_function(test[10:])
pdf=clf.decision_function(test)


# ## Probability Density Function of Real Signatures

# In[377]:

pd.DataFrame(pdf1).plot(kind="density",figsize=(5,5))
plt.show()


# ## Probability Density Function of Forged Signatures

# In[337]:

pd.DataFrame(pdf2).plot(kind="density",figsize=(5,5))
plt.show()


# In[338]:

def AER(preds):
    ac=preds[0:10]
    FRR=len(ac[ac==-1])/len(ac)
    fg=preds[10:]
    FAR=len(fg[fg==1])/len(fg)
    AER1=(FRR+FAR)/2
    return round(AER1*100,3)

print("The average error rate with hard thresholding is ", AER(preds))
print(preds)


# Clearly, given the small number of samples, the use of a hard threshold reduces the accuracy by a large margin, as shown in the paper. If we opt for a soft threshold, the corresponding error rate will fall.

# In[314]:

def AER2(preds):
    ac=preds[0:10]
    FRR=ac.count(-1)/len(ac)
    fg=preds[10:]
    FAR=fg.count(1)/len(fg)
    AER1=(FRR+FAR)/2
    return round(AER1*100,3),FRR,FAR


# In[384]:

def soft_thres(pdf,pdf1):
    m=np.mean(pdf1)
    sig=np.std(pdf1)
    best_thres=0
    best_AER=100
    best_FFR=100
    best_FAR=100
    for k in range(-300,300):
        k_d=k/100
        thres=m+k_d*sig
        preds2=[1 if x >= thres else -1 for x in pdf]
        #print(preds2)
        cur_AER,FRR,FAR=AER2(preds2)
       # print(k_d)
        if cur_AER < best_AER:
            best_AER=cur_AER
            best_thres=thres
            best_FRR=FRR
            best_FAR=FAR
           # print(best_AER,FRR,FAR)
    #print(f"The lowest possible error rate is {best_AER} at threshold = {best_thres}")
    return best_thres,best_FRR,best_FAR,best_AER




# Thus, after tuning the threshold, the average error rate has dropped to 10%.

# In[429]:

def fixed_soft_thres(pdf,thres):
    preds2=[1 if x >= thres else -1 for x in pdf]
    cur_AER,FRR,FAR=AER2(preds2)
    return cur_AER,FRR,FAR


# ## Aggregating data for 30 writers 

# In[358]:

24*42


# In[395]:

import random
li=random.sample(range(0,55),30)
li


# In[396]:

master_pdf1=np.ndarray(1)
master_pdf2=np.ndarray(1)
master_data=[]
for i in li:
    os=24*i
    #print(i)
    data=alt_df.iloc[0+os:14+os]
    #print(data.shape)
    data=data.drop("Writer_no",axis=1)
    data=data.drop("Sample_no",axis=1)
    
    li1=[os+j for j in range(0,24)]
  
    test_m=alt_df.drop(alt_df.index[li1])
    rand_signs=test_m.sample(n=20)
    
    test=alt_df.iloc[14+os:24+os]
    test=test.append(rand_signs)
    test=test.drop("Writer_no",axis=1)
    test=test.drop("Sample_no",axis=1)
    
    best_nu,best_gamma=tuning_SVM(data)
    clf = svm.OneClassSVM(nu=best_nu, kernel="rbf", gamma=best_gamma)
    clf.fit(data)
    preds=clf.predict(test)
    
    aer_h=AER(preds)

    pdf1=clf.decision_function(test[0:10])
    pdf2=clf.decision_function(test[10:])
    pdf=clf.decision_function(test)
    master_pdf1=np.append(master_pdf1,pdf1)
    master_pdf2=np.append(master_pdf2,pdf2)
    soft_t,frr,far,aer_s=soft_thres(pdf,pdf1)
    #print(master_pdf1)
    #print(f"The lowest possible error rate is {aer_s} at threshold = {soft_t}")
    x=pd.Series([i+1,best_nu,best_gamma,aer_h,soft_t,aer_s,frr,far],index=
                    ["Writer_no","Best_nu","Best_gamma","Hard_AER","Soft_Threshold","Soft_AER","FRR","FAR"])
    master_data.append(x)
    
master_df=pd.DataFrame(master_data)


# ## Dataset of 30 writers

# In[400]:

master_df


# In[404]:

print("Average AER with hard thresholding is ",np.mean(master_df.Hard_AER))
print("Average AER with soft thresholding is ",np.mean(master_df.Soft_AER))
print("Average soft threshold is ",np.mean(master_df.Soft_Threshold))


# ## Testing above generated numbers of remaining 25 writers
# ### Using average parameter and threshold value

# In[419]:

nu=np.mean(master_df.Best_nu)
gamma=np.mean(master_df.Best_gamma)
thres=np.mean(master_df.Soft_Threshold)


# In[426]:

li2=[]
for i in range(0,55):
    if i not in li:
        li2.append(i)


# In[430]:

testing_pdf1=np.ndarray(1)
testing_pdf2=np.ndarray(1)
testing_data=[]
for i in li2:
    os=24*i
    #print(i)
    data=alt_df.iloc[0+os:14+os]
    #print(data.shape)
    data=data.drop("Writer_no",axis=1)
    data=data.drop("Sample_no",axis=1)
    
    li1=[os+j for j in range(0,24)]
  
    test_m=alt_df.drop(alt_df.index[li1])
    rand_signs=test_m.sample(n=20)
    
    test=alt_df.iloc[14+os:24+os]
    test=test.append(rand_signs)
    test=test.drop("Writer_no",axis=1)
    test=test.drop("Sample_no",axis=1)
    

    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
    clf.fit(data)
    preds=clf.predict(test)
    
    aer_h=AER(preds)

    pdf1=clf.decision_function(test[0:10])
    pdf2=clf.decision_function(test[10:])
    pdf=clf.decision_function(test)
    testing_pdf1=np.append(testing_pdf1,pdf1)
    testing_pdf2=np.append(testing_pdf2,pdf2)
    
    aer_s,frr,far=fixed_soft_thres(pdf,thres)
    

    x=pd.Series([i+1,aer_h,aer_s,frr,far],index=
                    ["Writer_no","Hard_AER","Soft_AER","FRR","FAR"])
    testing_data.append(x)
    
testing_df=pd.DataFrame(testing_data)


# In[431]:

testing_df


# In[432]:

print("Average AER with hard thresholding is ",np.mean(testing_df.Hard_AER))
print("Average AER with soft thresholding is ",np.mean(testing_df.Soft_AER))


# ### The results obtained are consistent with the AERs in Guerbai et al. (2015). They report AER of between 21.04% to 32% for hard thresholding, and between 8.50% to 11.75% with soft thresholding.

# ## Probability Distribution Function of Real Signatures (Euclidean Distance Measure)

# In[436]:


pd.DataFrame(testing_pdf1).plot(kind="density",figsize=(5,5))
plt.show()


# ## Probability Distribution Function of Forged Signatures (Euclidean Distance Measure)

# In[439]:


pd.DataFrame(testing_pdf2).plot(kind="density",figsize=(5,5))
plt.show()


# In[ ]:




# In[ ]:




# # Trial work, discarded

# In[9]:

import pywt


# In[ ]:

r = cv2.selectROI(img)
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
cv2.imshow("Image", imCrop)
cv2.waitKey(0)


# In[98]:

get_ipython().magic('pinfo cv2.erode')


# In[100]:

get_ipython().magic('pinfo svm.OneClassSVM.fit')

