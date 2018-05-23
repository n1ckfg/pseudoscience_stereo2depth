# Josh Gladstone made this in 2018
# But a lot of it came from Timotheos Samartzidis
# http://timosam.com/python_opencv_depthimage
# Enjoy!

import numpy as np
import cv2
import sys, os
import glob
import time
import urllib.request
import unicodedata
from threading import *
from queue import *
from os.path import join
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox

def Calculate():
    global currentfile, img, height, width, imgL, imgR, titleStr
    if (currentfile != ''):
        settings.title(titleStr + '    ( Working. . . )')
        minDisparities=16
        window_size = w2.get()                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        rez = w0.get() / 20.0
        if (rez > 0):
            resL = cv2.resize(imgL,None,fx=rez, fy=rez, interpolation = cv2.INTER_AREA)
            resR = cv2.resize(imgR,None,fx=rez, fy=rez, interpolation = cv2.INTER_AREA)

            left_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=minDisparities,             # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize= w1.get(),
                P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=0,
                speckleRange=2,
                preFilterCap= w3.get(),
                mode=cv2.STEREO_SGBM_MODE_HH
            )
             
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
             
            lmbda = w4.get() * 1000
            sigma = 1.2
            visual_multiplier = 1
             
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)

            displ = left_matcher.compute(resL, resR)
            dispr = right_matcher.compute(resR, resL)
            imgLb = cv2.copyMakeBorder(imgL, top=0, bottom=0, left=np.uint16(minDisparities), right=0, borderType= cv2.BORDER_CONSTANT, value=[155,155,155] )
            filteredImg = wls_filter.filter(displ, imgLb, None, dispr)
            filteredImg = filteredImg * rez
            filteredImg = filteredImg + (w5.get()-100)
            filteredImg = (w6.get()/10.0)*(filteredImg - 128) + 128
            filteredImg = np.clip(filteredImg, 0, 255)
            filteredImg = np.uint8(filteredImg)
            if (precisealignmenthack == '0'):
                filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)     # Disparity truncation hack
                filteredImg = filteredImg[0:height, np.uint16(minDisparities/rez):width]                         #
                filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)     # Disparity truncation hack
            else:
                imgL2 = cv2.flip(imgL, 1)
                imgR2 = cv2.flip(imgR, 1)
                resL2 = cv2.flip(resL, 1)
                resR2 = cv2.flip(resR, 1)

                left_matcher2 = cv2.StereoSGBM_create(                                                           # Another disparity truncation hack
                    minDisparity=0,
                    numDisparities=minDisparities,             # max_disp has to be dividable by 16 f. E. HH 192, 256
                    blockSize= w1.get(),
                    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                    P2=32 * 3 * window_size ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=15,
                    speckleWindowSize=0,
                    speckleRange=2,
                    preFilterCap= w3.get(),
                    mode=cv2.STEREO_SGBM_MODE_HH
                )
             
                right_matcher2 = cv2.ximgproc.createRightMatcher(left_matcher2)
                 
                wls_filter2 = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher2)
                wls_filter2.setLambda(lmbda)
                wls_filter2.setSigmaColor(sigma)

                displ2 = left_matcher2.compute(resR2, resL2)
                dispr2 = right_matcher2.compute(resL2, resR2)
                imgLb2 = cv2.copyMakeBorder(imgL2, top=0, bottom=0, left=np.uint16(minDisparities), right=0, borderType= cv2.BORDER_CONSTANT, value=[155,155,155] )
                filteredImg2 = wls_filter.filter(displ2, imgLb2, None, dispr2)
                filteredImg2 = filteredImg2 * rez
                filteredImg2 = filteredImg2 + (w5.get()-100)
                filteredImg2 = (w6.get()/10.0)*(filteredImg2 - 128) + 128
                filteredImg2 = np.clip(filteredImg2, 0, 255)
                filteredImg2 = np.uint8(filteredImg2)
                filteredImg2 = cv2.flip(filteredImg2, 1)
                M = np.float32([[1,0,-16],[0,1,0]])
                filteredImg = cv2.warpAffine(filteredImg, M, (width, int(height/2)), borderMode=cv2.BORDER_WRAP)
                filteredImg2 = cv2.warpAffine(filteredImg2, M, (width, int(height/2)), borderMode=cv2.BORDER_WRAP)
                filteredImg2 = filteredImg2[0:height, 0:int(width/10)] 
                filteredImg = filteredImg[0:height, int(width/10):width]
                filteredImg = np.concatenate((filteredImg2, filteredImg), axis=1)
                
                # rows,cols = filteredImg.shape[:2]
                # gradientwidth = rows/10
                # for i in range(rows):
                #     for j in range(cols):
                #         if (j > (cols/2)-gradientwidth):
                #             alpha = np.clip((j - (cols/2)+gradientwidth)/gradientwidth, 0, 1)
                #         else:
                #             alpha = 0
                #         filteredImg[i][j] = alpha*filteredImg[i][j]+(1-alpha)*filteredImg2[i][j]
            filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)
            cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth Map', filteredImg)
            settings.title(titleStr)
            return filteredImg
        else:
            print ('Resolution must be greater than 0.')


def ThreadedCalculate(q, w0val, w1val, w2val, w3val, w4val, w5val, w6val, savefile):
    while True:
        img, filename = q.get()
        height, width = img.shape[:2]
        imgL = img[0:int((height/2)), 0:width]
        imgR = img[int((height/2)):height, 0:width]

        minDisparities=16
        window_size = w2val                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        rez = w0val / 20.0
        if (rez > 0):
            resL = cv2.resize(imgL,None,fx=rez, fy=rez, interpolation = cv2.INTER_AREA)
            resR = cv2.resize(imgR,None,fx=rez, fy=rez, interpolation = cv2.INTER_AREA)

            left_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=minDisparities,             # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize= w1val,
                P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=0,
                speckleRange=2,
                preFilterCap= w3val,
                mode=cv2.STEREO_SGBM_MODE_HH
            )
             
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
             
            lmbda = w4val * 1000
            sigma = 1.2
            visual_multiplier = 1
             
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)

            displ = left_matcher.compute(resL, resR)
            dispr = right_matcher.compute(resR, resL)
            imgLb = cv2.copyMakeBorder(imgL, top=0, bottom=0, left=np.uint16(minDisparities), right=0, borderType= cv2.BORDER_CONSTANT, value=[155,155,155] )
            filteredImg = wls_filter.filter(displ, imgLb, None, dispr)
            filteredImg = filteredImg * rez
            filteredImg = filteredImg + (w5val-100)
            filteredImg = (w6val/10.0)*(filteredImg - 128) + 128
            filteredImg = np.clip(filteredImg, 0, 255)
            filteredImg = np.uint8(filteredImg)
            if (precisealignmenthack == '0'):
                filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)     # Disparity truncation hack
                filteredImg = filteredImg[0:height, np.uint16(minDisparities/rez):width]                         #
                filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)     # Disparity truncation hack
            else:
                imgL2 = cv2.flip(imgL, 1)
                imgR2 = cv2.flip(imgR, 1)
                resL2 = cv2.flip(resL, 1)
                resR2 = cv2.flip(resR, 1)

                left_matcher2 = cv2.StereoSGBM_create(                                                           # Another disparity truncation hack
                    minDisparity=0,
                    numDisparities=minDisparities,             # max_disp has to be dividable by 16 f. E. HH 192, 256
                    blockSize= w1val,
                    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                    P2=32 * 3 * window_size ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=15,
                    speckleWindowSize=0,
                    speckleRange=2,
                    preFilterCap= w3val,
                    mode=cv2.STEREO_SGBM_MODE_HH
                )
             
                right_matcher2 = cv2.ximgproc.createRightMatcher(left_matcher2)
                 
                wls_filter2 = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher2)
                wls_filter2.setLambda(lmbda)
                wls_filter2.setSigmaColor(sigma)

                displ2 = left_matcher2.compute(resR2, resL2)
                dispr2 = right_matcher2.compute(resL2, resR2)
                imgLb2 = cv2.copyMakeBorder(imgL2, top=0, bottom=0, left=np.uint16(minDisparities), right=0, borderType= cv2.BORDER_CONSTANT, value=[155,155,155] )
                filteredImg2 = wls_filter.filter(displ2, imgLb2, None, dispr2)
                filteredImg2 = filteredImg2 * rez
                filteredImg2 = filteredImg2 + (w5val-100)
                filteredImg2 = (w6val/10.0)*(filteredImg2 - 128) + 128
                filteredImg2 = np.clip(filteredImg2, 0, 255)
                filteredImg2 = np.uint8(filteredImg2)
                filteredImg2 = cv2.flip(filteredImg2, 1)
                M = np.float32([[1,0,-16],[0,1,0]])
                filteredImg = cv2.warpAffine(filteredImg, M, (width, int(height/2)), borderMode=cv2.BORDER_WRAP)
                filteredImg2 = cv2.warpAffine(filteredImg2, M, (width, int(height/2)), borderMode=cv2.BORDER_WRAP)
                filteredImg2 = filteredImg2[0:height, 0:int(width/10)] 
                filteredImg = filteredImg[0:height, int(width/10):width]
                filteredImg = np.concatenate((filteredImg2, filteredImg), axis=1)
                
            #filteredImg = cv2.resize(filteredImg,(width,int(height/2)), interpolation = cv2.INTER_CUBIC)

            dispthread = Thread(target=threadDisplay, args=(filteredImg,imgL,imgR))
            dispthread.start()
            if (savefile == 1):
                if (savefiletype == 'JPEG'):
                    cv2.imwrite(batchpathname + '/' + filename + '.jpg', filteredImg, [cv2.IMWRITE_JPEG_QUALITY, jpegquality]) 
                elif (savefiletype == 'PNG'):
                    cv2.imwrite(batchpathname + '/' + filename + '.png', filteredImg)
                elif (savefiletype == 'TIFF'):
                    cv2.imwrite(batchpathname + '/' + filename + '.tif', filteredImg)
            elif (savefile == 2):
                filteredImg = cv2.cvtColor(filteredImg, cv2.COLOR_GRAY2RGB)
                dof = np.concatenate((imgL, filteredImg), axis=0)
                if (savefiletype == 'JPEG'):
                    cv2.imwrite(batchpathname + '/' + filename + '.jpg', dof, [cv2.IMWRITE_JPEG_QUALITY, jpegquality]) 
                elif (savefiletype == 'PNG'):
                    cv2.imwrite(batchpathname + '/' + filename + '.png', dof)
                elif (savefiletype == 'TIFF'):
                    cv2.imwrite(batchpathname + '/' + filename + '.tif', dof)        
        else:
            print ('Resolution must be greater than 0.')
        q.task_done()

def openfile():
    global currentfile, currentdirectory, img, height, width, imgL, imgR, titleStr, pathname, filename, files, currentfiletype, seekwindow, seekslider, framecount, InFrame, OutFrame, setInText, setOutText, durationtText
    del files[:]
    currentdirectory = ''
    currentfile = filedialog.askopenfilename()
    pathname = os.path.dirname(currentfile)
    exttype = os.path.splitext(os.path.basename(currentfile))[1]
    if (currentfile != ''):
        if (exttype == '.mp4' or exttype == '.mov' or exttype == '.webm'):
            currentfiletype = 'video'
            cap = cv2.VideoCapture(currentfile)
            ret,img = cap.read()
            framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            height, width = img.shape[:2]
            imgL = img[0:int((height/2)), 0:width]
            imgR = img[int((height/2)):height, 0:width]
            cv2.namedWindow('Left Source', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Right Source', cv2.WINDOW_NORMAL)
            cv2.imshow('Left Source', imgL)
            cv2.imshow('Right Source', imgR)
            titleStr = 'Stereo2Depth   [Batching ' + str(framecount) + ' frames]'
            Calculate()
            try:
                seekwindow.deiconify()
            except:
                seekwindow = Tk()
                seekwindow.title('Seek')
                seekwindow.geometry('520x80+720+660')
                seekslider = Scale(seekwindow, from_=1, to=100, orient=HORIZONTAL, length=500)
                seekslider.bind('<ButtonRelease-1>', updateValue)
                seekslider.grid(row=0,column=0,padx=5)
                InCanvas = Canvas(seekwindow)
                InCanvas.grid(row=1, column=0, padx=6, pady=8, sticky=W)
                setInButton = Button(InCanvas, text='Set In', width=5, command=lambda:setinout(True))
                setInButton.grid(row=0,column=0, sticky=W)
                setInButton.configure(background='white')
                setInText = Label(InCanvas, text='In Frame:  ')
                setInText.grid(row=0,column=1,padx=10,sticky=W)
                OutCanvas = Canvas(seekwindow)
                OutCanvas.grid(row=1, column=0, padx=6, pady=8, sticky=E)
                setOutText = Label(OutCanvas, text='Out Frame:  ')
                setOutText.grid(row=0,column=2, padx=10)
                setOutButton = Button(OutCanvas, text='Set Out', width=5, command=lambda:setinout(False))
                setOutButton.grid(row=0,column=3)
                setOutButton.configure(background='white')
                durationtText = Label(seekwindow, justify=CENTER, text=' ')
                durationtText.grid(row=1,column=0, padx=10)
            seekslider = Scale(seekwindow, from_=1, to=framecount, orient=HORIZONTAL, length=500)
            seekslider.bind('<ButtonRelease-1>', updateValue)
            seekslider.grid(row=0,column=0,padx=5)
            seekslider.set(1)
            InFrame = 1
            OutFrame = framecount
            setInText.config(text='In Frame: ' + str(InFrame))
            setOutText.config(text='Out Frame: ' + str(OutFrame))
            durationtText.config(text=str(OutFrame - InFrame + 1) + ' frames')
        elif (exttype == '.jpg' or exttype == '.jpeg' or exttype == '.png'):
            currentfiletype = 'image'
            filename = os.path.splitext(os.path.basename(currentfile))[0]
            img = cv2.imread(currentfile)
            height, width = img.shape[:2]
            imgL = img[0:int((height/2)), 0:width]
            imgR = img[int((height/2)):height, 0:width]
            cv2.namedWindow('Left Source', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Right Source', cv2.WINDOW_NORMAL)
            cv2.imshow('Left Source', imgL)
            cv2.imshow('Right Source', imgR)
            titleStr = 'Stereo2Depth'
            
            seekwindow.withdraw()
            Calculate()
        else:
            print ('Unrecognized file type')

def openfolder():
    global currentfile, currentdirectory, img, height, width, imgL, imgR, titleStr, pathname, filename, files, currentfiletype, seekwindow, seekslider, InFrame, OutFrame, setInText, setOutText, durationtText
    del files[:]
    currentfiletype = 'image'
    currentdirectory = filedialog.askdirectory()
    pathname = currentdirectory
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
    for extension in extensions:
        files.extend(glob.glob(currentdirectory + '/' + extension))
    print ('Batching ' + str(len(files)) + ' frames')
    if (len(files) > 0):
        currentfile = files[0]
        filename = os.path.splitext(os.path.basename(currentfile))[0]
        img = cv2.imread(currentfile)
        height, width = img.shape[:2]
        imgL = img[0:int((height/2)), 0:width]
        imgR = img[int((height/2)):height, 0:width]
        cv2.namedWindow('Left Source', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Source', cv2.WINDOW_NORMAL)
        cv2.imshow('Left Source', imgL)
        cv2.imshow('Right Source', imgR)
        titleStr = 'Stereo2Depth   [Batching ' + str(len(files)) + ' files]'
        Calculate()
        try:
            seekwindow.deiconify()
        except:
            seekwindow = Tk()
            seekwindow.title('Seek')
            seekwindow.geometry('520x80+720+660')
            seekslider = Scale(seekwindow, from_=1, to=100, orient=HORIZONTAL, length=500)
            seekslider.bind('<ButtonRelease-1>', updateValue)
            seekslider.grid(row=0,column=0,padx=5)
            InCanvas = Canvas(seekwindow)
            InCanvas.grid(row=1, column=0, padx=6, pady=8, sticky=W)
            setInButton = Button(InCanvas, text='Set In', width=5, command=lambda:setinout(True))
            setInButton.grid(row=0,column=0, sticky=W)
            setInButton.configure(background='white')
            setInText = Label(InCanvas, text='In Frame:  ')
            setInText.grid(row=0,column=1,padx=10,sticky=W)
            OutCanvas = Canvas(seekwindow)
            OutCanvas.grid(row=1, column=0, padx=6, pady=8, sticky=E)
            setOutText = Label(OutCanvas, text='Out Frame:  ')
            setOutText.grid(row=0,column=2, padx=10)
            setOutButton = Button(OutCanvas, text='Set Out', width=5, command=lambda:setinout(False))
            setOutButton.grid(row=0,column=3)
            setOutButton.configure(background='white')
            durationtText = Label(seekwindow, justify=CENTER, text=' ')
            durationtText.grid(row=1,column=0, padx=10)
        seekslider = Scale(seekwindow, from_=1, to=len(files), orient=HORIZONTAL, length=500)
        seekslider.bind('<ButtonRelease-1>', updateValue)
        seekslider.grid(row=0,column=0,padx=5)
        seekslider.set(1)
        InFrame = 1
        OutFrame = len(files)
        setInText.config(text='In Frame: ' + str(InFrame))
        setOutText.config(text='Out Frame: ' + str(OutFrame))
        durationtText.config(text=str(OutFrame - InFrame + 1) + ' frames')
    else:
        titleStr = 'Stereo2Depth'


def updateValue(event):
    global seekslider, currentfile, framecount, img, height, width, imgL, imgR, currentfiletype, files
    if (currentfiletype == 'video'):
        cap = cv2.VideoCapture(currentfile)
        cap.set(cv2.CAP_PROP_POS_FRAMES,seekslider.get()-1);
        ret, img = cap.read()
    elif (currentfiletype == 'image'):
        img = cv2.imread(files[seekslider.get()-1])
    else:
        return
    height, width = img.shape[:2]
    imgL = img[0:int((height/2)), 0:width]
    imgR = img[int((height/2)):height, 0:width]
    cv2.imshow('Left Source', imgL)
    cv2.imshow('Right Source', imgR)
    Calculate()

def threadDisplay(depthmap, imgl, imgr):
    try:
        cv2.imshow('Depth Map', depthmap)
        cv2.imshow('Left Source', imgl)
        cv2.imshow('Right Source', imgr)
    except:
        pass

def SaveFile(savefile, batch):
    global currentfile, currentdirectory, img, height, width, imgL, imgR, pathname, filename, files, batchpathname, currentfiletype, abort, InFrame, OutFrame
    
    if (batch == 0 and savefile != 0 and currentfile != ''):
        filename = os.path.splitext(os.path.basename(currentfile))[0]
        thedepth = Calculate()

        if (savefile == 1):
            if (savefiletype == 'JPEG'):
                cv2.imwrite(pathname + '/' + filename + '_depthmap.jpg', thedepth, [cv2.IMWRITE_JPEG_QUALITY, jpegquality])
                print ('Saved: ' + pathname + '/' + filename + '_depthmap.jpg\a')
            elif (savefiletype == 'PNG'):
                cv2.imwrite(pathname + '/' + filename + '_depthmap.png', thedepth)
                print ('Saved: ' + pathname + '/' + filename + '_depthmap.pngs\a')
            elif (savefiletype == 'TIFF'):
                cv2.imwrite(pathname + '/' + filename + '_depthmap.tif', thedepth)
                print ('Saved: ' + pathname + '/' + filename + '_depthmap.tif\a')
        elif (savefile == 2):
            thedepth = cv2.cvtColor(thedepth, cv2.COLOR_GRAY2RGB)
            dof = np.concatenate((imgL, thedepth), axis=0)
            if (savefiletype == 'JPEG'):
                cv2.imwrite(pathname + '/' + filename + '_6DoF.jpg', dof, [cv2.IMWRITE_JPEG_QUALITY, jpegquality])
                print ('Saved: ' + pathname + '/' + filename + '_6DoF.jpg\a')
            elif (savefiletype == 'PNG'):
                cv2.imwrite(pathname + '/' + filename + '_6DoF.png', dof)
                print ('Saved: ' + pathname + '/' + filename + '_6DoF.png\a')
            elif (savefiletype == 'TIFF'):
                cv2.imwrite(pathname + '/' + filename + '_6DoF.tif', dof)
                print ('Saved: ' + pathname + '/' + filename + '_6DoF.tif\a')

    elif (batch == 0):
        print ('No file loaded')
    if (batch == 1 and len(files) >= 1):
        try:
            progresswindow.deiconify()
        except:
            progresswindow = Tk()
            progresswindow.title('Progress')
            if (os.name == 'nt'):
                progresswindow.geometry('520x110+720+660')
            else:
                progresswindow.geometry('520x100+720+660')
            progressText = Label(progresswindow,justify=CENTER, text='File (5/10)  --  50%')
            progressText.grid(row=0,column=0,padx=5,pady=5)
            progressBar = ttk.Progressbar(progresswindow, orient='horizontal', length=500, mode='determinate')
            progressBar.grid(row=1,column=0,padx=10, pady=10)
            cancelButton = Button(progresswindow, text='Cancel Batch Export', width=40, command=cancelsave)
            cancelButton.grid(row=2,column=0,padx=10, pady=5)
        filename = os.path.splitext(os.path.basename(files[0]))[0]
        #filename = re.sub(r'\W+', '', filename)
        filename = str(unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore'))
        filename = filename.lstrip('b')
        filename = filename.strip('\'')
        if (savefile == 1):
            batchpathname = pathname + '/' + filename + '_depth'
        else:
            batchpathname = pathname + '/' + filename + '_6DoF'
        if not os.path.exists(batchpathname):
            os.makedirs(batchpathname)
        print ('Saving to: ' + batchpathname + '/')
        starttime = time.time()
        q = Queue(maxsize=0)

        for i in range(numberofthreads):
            worker = Thread(target=ThreadedCalculate, args=(q, w0.get(), w1.get(), w2.get(), w3.get(), w4.get(), w5.get(), w6.get(), savefile))
            worker.setDaemon(True)
            worker.start()

        index = InFrame-1
        abort=False
        settings.title(titleStr + '    ( Working. . . )')
        lastSave = time.time()

        while (index < OutFrame):
            currentfile = files[index]
            filename = os.path.splitext(os.path.basename(currentfile))[0]
            img = cv2.imread(currentfile)
            filename = str(unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore'))
            filename = filename.lstrip('b')
            filename = filename.strip('\'')
            q.put((img, filename))
            # print ('%0.2f' % (100 * (index) / len(files)) + '%') 
            timeperframe = time.time()-lastSave
            progressText.config(text = 'File (' + str(index-InFrame+1) + '/' + str((OutFrame-InFrame)+1) + ')   --   ' + '%0.2f' % (100 * ((index-InFrame+1) / (OutFrame-InFrame+1))) + '%   --   ' + '%0.2f' % (timeperframe * ((OutFrame-InFrame+1) - (index - InFrame)) / 60) + ' minutes left')
            progressBar['value'] = 100 * ((index-InFrame+1) / (OutFrame-InFrame+1))
            progresswindow.update()

            k = cv2.waitKey(1) 
            if (k==27):    # Esc key to stop
                abort = True
                break  
            if (abort):
                break
            index = index + 1
            lastSave = time.time()
            q.join()

        settings.title(titleStr)
        progresswindow.withdraw()
        if not abort:
            print ('Batch export complete in ' + '%0.2f' % (time.time() - starttime) + ' seconds.\a')
        else:
            print('Batch export aborted after ' + '%0.2f' % (time.time() - starttime) + ' seconds.')
    elif (batch == 1 and currentfiletype == 'video'):
        try:
            progresswindow.deiconify()
        except:
            progresswindow = Tk()
            progresswindow.title('Progress')
            if (os.name == 'nt'):
                progresswindow.geometry('520x110+720+660')
            else:
                progresswindow.geometry('520x100+720+660')
            progressText = Label(progresswindow,justify=CENTER, text='File (5/10)  --  50%')
            progressText.grid(row=0,column=0,padx=5,pady=5)
            progressBar = ttk.Progressbar(progresswindow, orient='horizontal', length=500, mode='determinate')
            progressBar.grid(row=1,column=0,padx=10, pady=10)
            cancelButton = Button(progresswindow, text='Cancel Batch Export', width=40, command=cancelsave)
            cancelButton.grid(row=2,column=0,padx=10, pady=5)
            filename = os.path.splitext(os.path.basename(currentfile))[0]
            # filename = re.sub(r'\W+', '', filename)
            filename = str(unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore'))
            filename = filename.lstrip('b')
            filename = filename.strip('\'')
        if (savefile == 1):
            batchpathname = pathname + '/' + filename + '_depth'
        else:
            batchpathname = pathname + '/' + filename + '_6DoF'
        if not os.path.exists(batchpathname):
            os.makedirs(batchpathname)
        print ('Saving to: ' + batchpathname + '/')

        starttime = time.time()
        q = Queue(maxsize=0)

        for i in range(numberofthreads):
            worker = Thread(target=ThreadedCalculate, args=(q, w0.get(), w1.get(), w2.get(), w3.get(), w4.get(), w5.get(), w6.get(), savefile))
            worker.daemon = True
            worker.start()

        index = InFrame-1
        cap = cv2.VideoCapture(currentfile)
        cap.set(cv2.CAP_PROP_POS_FRAMES,InFrame-1);
        filenamebase = os.path.splitext(os.path.basename(currentfile))[0]
        numberofdigits = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        abort=False
        lastSave = time.time()

        while (index < OutFrame):
            settings.title(titleStr + '    ( Working. . . )')
            ret, img = cap.read()
            filename = filenamebase + '_' + str(index+1).zfill(numberofdigits)
            filename = str(unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore'))
            filename = filename.lstrip('b')
            filename = filename.strip('\'')
            q.put((img, filename))
            # print ('%0.2f' % (100 * (index) / framecount) + '%')
            timeperframe = time.time()-lastSave
            progressText.config(text = 'File (' + str(index-InFrame+1) + '/' + str((OutFrame-InFrame)+1) + ')   --   ' + '%0.2f' % (100 * ((index-InFrame+1) / (OutFrame-InFrame+1))) + '%   --   ' + '%0.2f' % (timeperframe * ((OutFrame-InFrame+1) - (index - InFrame)) / 60) + ' minutes left')
            progressBar['value'] = 100 * ((index-InFrame+1) / (OutFrame-InFrame+1))
            progresswindow.update()

            k = cv2.waitKey(1) 
            if (k==27):    # Esc key to stop
                abort = True
                break  
            if (abort):
                break
            index = index + 1
            lastSave = time.time()
            q.join()

        settings.title(titleStr)
        progresswindow.withdraw()
        if not abort:
            print ('Batch export complete in ' + '%0.2f' % (time.time() - starttime) + ' seconds.\a')
        else:
            print('Batch export aborted after ' + '%0.2f' % (time.time() - starttime) + ' seconds.')
    elif (batch == 1):
        print ('No batch loaded')

def autoupdate(value):
    global autoupdatebool
    if (autoupdatebool.get() == 1):
        Calculate()

def cancelsave():
    global abort
    abort = True

def defaultSettings():
    w0.set(4)
    w1.set(5)
    w2.set(5)
    w3.set(60)
    w4.set(80)
    w5.set(100)
    w6.set(10)
    settings.update()
    Calculate()

def advancedsettings(event):
    global threadsslider, savefiletypestring, jpegqualityslider, precisealignmentbool, numberofthreads, savefiletype, jpegquality, precisealignmenthack
    try:
        numberofthreads = threadsslider.get()
        savefiletype = savefiletypestring.get()
        jpegquality = jpegqualityslider.get()
        precisealignmenthack = precisealignmentbool.get()
    except:
        pass

def showadvancedsettings():
    global advancedsettingswindow, threadsslider, savefiletypestring, precisealignmentbool, jpegqualityslider, numberofthreads, savefiletype, jpegquality, precisealignmenthack
    try:
        advancedsettingswindow.destroy()
    except:
        pass

    advancedsettingswindow = Tk()
    advancedsettingswindow.title('Advanced Settings')
    advancedsettingswindow.geometry('450x220+85+350')
    advancedsettingsCanvas = Canvas(advancedsettingswindow)
    advancedsettingsCanvas.grid(row=0, column=0, padx=40, pady=15)
    Label(advancedsettingsCanvas, text='Number of Threads').grid(row=0,column=0,padx=5,sticky=E)
    threadsslider = Scale(advancedsettingsCanvas, from_=1, to=100, orient=HORIZONTAL, length=200, command=advancedsettings)
    threadsslider.grid(row=0,column=1,padx=5)
    threadsslider.set(numberofthreads)
    Label(advancedsettingsCanvas, text='Save File Type').grid(row=1,column=0,padx=5,sticky=E)
    savefiletypestring = StringVar(advancedsettingsCanvas)
    savefiletypestring.set(savefiletype) # default value
    savefiletypedropdown = OptionMenu(advancedsettingsCanvas, savefiletypestring, 'JPEG', 'PNG', 'TIFF', command=advancedsettings)
    savefiletypedropdown.config(width=15)
    savefiletypedropdown.grid(row=1,column=1,pady=15)
    Label(advancedsettingsCanvas, text='Jpeg Quality').grid(row=2,column=0,padx=5,sticky=E)
    jpegqualityslider = Scale(advancedsettingsCanvas, from_=1, to=100, orient=HORIZONTAL, length=200, command=advancedsettings)
    jpegqualityslider.grid(row=2,column=1,padx=5)
    jpegqualityslider.set(jpegquality)
    Label(advancedsettingsCanvas, text='Precise Alignment Hack\n(doubles processing time)').grid(row=3,column=0,padx=5,sticky=E)
    precisealignmentbool = StringVar(advancedsettingsCanvas)  
    precisealignmentbool.set(precisealignmenthack)    
    precisealignmentcheck = Checkbutton(advancedsettingsCanvas, variable=precisealignmentbool, command=lambda:advancedsettings(0))
    #precisealignmentcheck.config(width=15)
    precisealignmentcheck.grid(row=3,column=1,pady=15,columnspan=2)

def seekthread(seekto, cap):
    x = 0
    while (x + 1 < seekto):
        cap.read()
        x = x + 1
        print ('Seeking -- ' + '%0.2f' % (100 * x / seekto) + '%')

def setinout(setin):
    global InFrame, OutFrame, setInText, setOutText, titleStr
    framenumber = seekslider.get()
    if (setin == True):
        setInText.config(text='In Frame: ' + str(framenumber))
        InFrame = framenumber
    else:
        setOutText.config(text='Out Frame: ' + str(framenumber))
        OutFrame = framenumber
    if (InFrame <= OutFrame):
        titleStr = 'Stereo2Depth   [Batching ' + str(OutFrame - InFrame + 1) + ' frames]'
        durationtText.config(text=str(OutFrame - InFrame + 1) + ' frames')
    else:
        titleStr = 'Stereo2Depth   [In/Out Frame Error]'
        durationtText.config(text='ERROR')
        print ('ERROR: In Frame must be before Out Frame')
    settings.title(titleStr)

def VersionCheck():
    try:
        with urllib.request.urlopen('http://pseudoscience.pictures/stereo2depth/latestversion.txt') as response:
            html = str(response.read())
            html = html.lstrip('b')
            html = html.strip('\'')
            theversion = sys.argv[0]
            theversion = theversion[-6:]
            theversion = theversion.rstrip('.py')
            print(theversion)
        if (html != theversion):
            print ('New version available! Check the sidebar at reddit.com/r/6DoF to download Stereo2Depth version ' + html)
            messagebox.showwarning(
                'Update Available',
                'New version available! Check the sidebar at reddit.com/r/6DoF to download Stereo2Depth version ' + html
            )
            return
    except:
        pass

currentfile = ''
currentfiletype = ''
titleStr = 'Stereo2Depth'
files = []
settings = Tk()
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth Map', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Depth Map', 800,400)
cv2.moveWindow('Depth Map', 580,225);
cv2.namedWindow('Left Source', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Left Source', 250,125)
cv2.moveWindow('Left Source', 580,65);
cv2.namedWindow('Right Source', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Right Source', 250,125)
cv2.moveWindow('Right Source', 830,65);
autoupdatebool = IntVar()
settings.title(titleStr)
if (os.name == 'nt'):
    settings.geometry('520x590+50+65')
else:
    settings.geometry('520x520+50+65')
settings.columnconfigure(0, weight=1)
settings.columnconfigure(1, weight=1)
seekwindow = Tk()
seekwindow.title('Seek')
seekwindow.geometry('520x80+720+660')
seekslider = Scale(seekwindow, from_=1, to=100, orient=HORIZONTAL, length=500)
seekslider.bind('<ButtonRelease-1>', updateValue)
seekslider.grid(row=0,column=0,padx=5)
InCanvas = Canvas(seekwindow)
InCanvas.grid(row=1, column=0, padx=6, pady=8, sticky=W)
setInButton = Button(InCanvas, text='Set In', width=5, command=lambda:setinout(True))
setInButton.grid(row=0,column=0, sticky=W)
setInButton.configure(background='white')
setInText = Label(InCanvas, text='In Frame:  ')
setInText.grid(row=0,column=1,padx=10,sticky=W)
OutCanvas = Canvas(seekwindow)
OutCanvas.grid(row=1, column=0, padx=6, pady=8, sticky=E)
setOutText = Label(OutCanvas, text='Out Frame:  ')
setOutText.grid(row=0,column=0, padx=10)
setOutButton = Button(OutCanvas, text='Set Out', width=5, command=lambda:setinout(False))
setOutButton.grid(row=0,column=1)
setOutButton.configure(background='white')
durationtText = Label(seekwindow, justify=CENTER, text=' ')
durationtText.grid(row=1,column=0, padx=10)
seekwindow.withdraw()

advancedsettingswindow = Tk()
advancedsettingswindow.title('Advanced Settings')
advancedsettingswindow.geometry('450x230+85+350')
advancedsettingsCanvas = Canvas(advancedsettingswindow)
advancedsettingsCanvas.grid(row=0, column=0, padx=40, pady=15)
Label(advancedsettingsCanvas, text='Number of Threads').grid(row=0,column=0,padx=5,sticky=E)
threadsslider = Scale(advancedsettingsCanvas, from_=1, to=100, orient=HORIZONTAL, length=200)
threadsslider.grid(row=0,column=1,padx=5)
threadsslider.set(20)
Label(advancedsettingsCanvas, text='Save File Type').grid(row=1,column=0,padx=5,sticky=E)
savefiletypestring = StringVar(advancedsettingsCanvas)
savefiletypestring.set('JPEG') # default value
savefiletypedropdown = OptionMenu(advancedsettingsCanvas, savefiletypestring, 'JPEG', 'PNG', 'TIFF')
savefiletypedropdown.config(width=15)
savefiletypedropdown.grid(row=1,column=1,pady=15)
Label(advancedsettingsCanvas, text='Jpeg Quality').grid(row=2,column=0,padx=5,sticky=E)
jpegqualityslider = Scale(advancedsettingsCanvas, from_=1, to=100, orient=HORIZONTAL, length=200)
jpegqualityslider.grid(row=2,column=1,padx=5)
jpegqualityslider.set(100)
Label(advancedsettingsCanvas, text='Precise Alignment Hack\n(doubles processing time)').grid(row=3,column=0,padx=5,sticky=E)
precisealignmentbool = StringVar(advancedsettingsCanvas)  
precisealignmentcheck = Checkbutton(advancedsettingsCanvas, variable=precisealignmentbool)
#precisealignmentcheck.config(width=15)
precisealignmentcheck.grid(row=3,column=1,pady=15)
advancedsettings(0)
advancedsettingswindow.withdraw()
numberofthreads = 20
savefiletype='JPEG'
jpegquality=100
precisealignmenthack = '0'

progresswindow = Tk()
progresswindow.title('Progress')
if (os.name == 'nt'):
    progresswindow.geometry('520x110+720+660')
else:
    progresswindow.geometry('520x100+720+660')
progressText = Label(progresswindow,justify=CENTER, text='File (5/10)  --  50%')
progressText.grid(row=0,column=0,padx=5,pady=5)
progressBar = ttk.Progressbar(progresswindow, orient='horizontal', length=500, mode='determinate')
progressBar.grid(row=1,column=0,padx=10, pady=10)
cancelButton = Button(progresswindow, text='Cancel Batch Export', width=40, command=cancelsave)
cancelButton.grid(row=2,column=0,padx=10, pady=5)
progresswindow.withdraw()

sliderCanvas = Canvas(settings)
sliderCanvas.grid(row=0, column=0, padx=5, pady=15, columnspan=2)

Label(sliderCanvas,justify=LEFT, text='Resolution').grid(row=0,column=0,padx=5,sticky=E)
w0 = Scale(sliderCanvas, from_=1, to=20, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w0.grid(row=0,column=1,padx=5,pady=7,sticky=W)
w0.set(4)
Label(sliderCanvas,justify=LEFT, text='Block Size').grid(row=1,column=0,padx=5,sticky=E)
w1 = Scale(sliderCanvas, from_=0, to=25, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w1.grid(row=1,column=1,padx=5,pady=7,sticky=W)
w1.set(5)
Label(sliderCanvas,justify=LEFT, text='Window Size').grid(row=2,column=0,padx=5,sticky=E)
w2 = Scale(sliderCanvas, from_=0, to=15, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w2.grid(row=2,column=1,padx=5,pady=7,sticky=W)
w2.set(5)
Label(sliderCanvas,justify=LEFT, text='Filter Cap').grid(row=3,column=0,padx=5,sticky=E)
w3 = Scale(sliderCanvas, from_=0, to=100, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w3.grid(row=3,column=1,padx=5,pady=7,sticky=W)
w3.set(60)
Label(sliderCanvas,justify=LEFT, text='Lmbda').grid(row=4,column=0,padx=5,sticky=E)
w4 = Scale(sliderCanvas, from_=0, to=100, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w4.grid(row=4,column=1,padx=5,pady=7,sticky=W)
w4.set(80)
Label(sliderCanvas,justify=LEFT, text='Brightness').grid(row=5,column=0,padx=5,sticky=E)
w5 = Scale(sliderCanvas, from_=0, to=200, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w5.grid(row=5,column=1,padx=5,pady=7,sticky=W)
w5.set(100)
Label(sliderCanvas,justify=LEFT, text='Contrast').grid(row=6,column=0,padx=5,sticky=E)
w6 = Scale(sliderCanvas, from_=0, to=30, orient=HORIZONTAL, length=350, showvalue=0, command=autoupdate)
w6.grid(row=6,column=1,padx=5,pady=7,sticky=W)
w6.set(10)
settings.update()

buttonCanvas = Canvas(settings)
buttonCanvas.grid(row=1, column=0, padx=10, columnspan=2)
updateCanvas = Canvas(buttonCanvas)
updateCanvas.grid(row=0, column=0, padx=0, pady=7, columnspan=2)
updateButton = Button(updateCanvas, text='Update', width=40, command=Calculate)
updateButton.grid(row=0,column=0, columnspan=2)
updateButton.configure(background='white')
checkbox = Checkbutton(updateCanvas, text='Auto-Update', variable=autoupdatebool)
checkbox.grid(row=1,column=0,columnspan=2)

openButton = Button(buttonCanvas, text='Open File', width=25, command=openfile)
openButton.grid(row=1,column=0,columnspan=2,pady=10,padx=20,sticky=W)
openButton.configure(background='white')

openbatchButton = Button(buttonCanvas, text='Open Directory', width=25, command=openfolder)
openbatchButton.grid(row=1,column=1,columnspan=2,pady=10,padx=20,sticky=E)
openbatchButton.configure(background='white')

saveButton = Button(buttonCanvas, text='Export Single Depth Map', width=25, command=lambda:SaveFile(1, 0))
saveButton.grid(row=2,column=0,pady=10,padx=20,sticky=E)
saveButton.configure(background='white')

savebatchButton = Button(buttonCanvas, text='Batch Export Depth Maps', width=25, command=lambda:SaveFile(1, 1))
savebatchButton.grid(row=2,column=1,pady=10,padx=20,sticky=E)
savebatchButton.configure(background='white')

saveButton = Button(buttonCanvas, text='Export Single 6DoF', width=25, command=lambda:SaveFile(2, 0))
saveButton.grid(row=3,column=0,pady=10,padx=20,sticky=W)
saveButton.configure(background='white')

savebatchButton = Button(buttonCanvas, text='Batch Export 6DoF', width=25, command=lambda:SaveFile(2, 1))
savebatchButton.grid(row=3,column=1,pady=10,padx=20,sticky=E)
savebatchButton.configure(background='white')

defaultsButton = Button(buttonCanvas, text='Reset Defaults', width=40, command=defaultSettings)
defaultsButton.grid(row=4,column=0,columnspan=2,pady=10,padx=20)
defaultsButton.configure(background='white')

advancedsettingsButton = Button(buttonCanvas, text='Advanced Settings', width=40, command=showadvancedsettings)
advancedsettingsButton.grid(row=5,column=0,columnspan=2,pady=10,padx=20)
advancedsettingsButton.configure(background='white')

VersionCheck()

mainloop()
