
import numpy as np
import cv2

#putting coordinates of square in specific order--NW,NE,SE,SW
def rectify(h):
    #reshaping h from 4x1x2 to 4x2
	h = h.reshape((4,2))
    #defining array hnew of 4x2 of datatype float
	hnew = np.zeros((4,2),dtype = np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]
	 
	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]

	return hnew


#list of all images for training    
images = ['training_images\sudoku.jpg', 'training_images\sudoku2.jpg', 'training_images\sudoku3.jpg', 'training_images\sudoku4.jpg',
        'training_images\sudoku5.jpg', 'training_images\sudoku6.jpg', 'training_images\sudoku7.jpg', 'training_images\sudoku7.jpg',
        'training_images\sudoku9.jpg']

#defining list for samples and responses        
samples =  np.empty((0, 1600))
responses = []

#iterating over each image
for idx,image in enumerate(images):
    
    #reading image
    img = cv2.imread(image)

    #converting image into grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #blurring image and taking adaptive threshold of grayscale image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    #getting all contours in contours array
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #finding contour with greatest area
    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        arc = cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i, 0.02*arc, True)
        if area > max_area and len(approx) == 4:
            #biggest holds coordinates of largest square which helps us to exract sudoku square
            biggest = approx
            max_area = area
    
    
    #getting coordinates in specific order
    sq_old = rectify(biggest)

    #creating array for transformation
    sq_new = np.array([ [0,0],[359,0],[359,359],[0,359] ],np.float32)

    #transforming extracted image into an image of 360x360 
    trans = cv2.getPerspectiveTransform(sq_old,sq_new)
    new_img = cv2.warpPerspective(gray,trans,(360,360))
    #displaing extracted image
    cv2.imshow('Extracted Image',new_img)
    cv2.waitKey(0)

    #blurring image and taking adaptive threshold of extracted image
    new_blur = cv2.GaussianBlur(new_img,(5,5),0)
    new_thresh = cv2.adaptiveThreshold(new_blur,255,1,1,11,2)

    
    # keyboard mappings for 0-9; user may type in this range when prompted
    keys = [i for i in range(48, 58)]

    #h: height  w: width    of the small square to be selected
    h = 40
    w = 40
    sm_new = np.array([ [0,0],[39,0],[39,39],[0,39] ],np.float32)
    for i in range(0,9):
        for j in range(0,9):
            #x and y are the top left coordinates of each small square
            x = j*40
            y = i*40
            #hl:horizontal left     hr:horizontal right     vu:vertical up  vd:vertical down
            #these are the adjusted coordinatesof small square
            #this is done to remove the lineson the edges
            hl = x + 2
            hr = x + w - 3
            vu = y + 2
            vd = y + h - 3
            
            #draw the bounding box on the image
            #cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            
            #roi: region of interest
            roi = new_thresh[vu:vd, hl:hr]
            #resizing roi
            roi_small = cv2.resize(roi, (40, 40))
            
            sm_old = np.array([[hl,vu],[hr,vu],[hr,vd],[hl,vd]],np.float32)
            #print(sm_old)
            
            #transforming small square into 40x40 to display
            sm_trans = cv2.getPerspectiveTransform(sm_old,sm_new)
            sm_warp = cv2.warpPerspective(new_img,sm_trans,(40,40))
            cv2.imshow('Square Pixel',sm_warp)
            print ("Enter digit for " + str(i) + "," + str(j)) 
            #waiting for user to enter the right value of small square
            key = cv2.waitKey(0)
            
            if key == 27:
                sys.exit()
            elif key in keys:
                # save pixel data in 1x1600 matrix of 'samples'
                sample = roi_small.reshape((1,1600))                
                samples = np.append(samples,sample,0)
                # save input in 'responses'
                responses.append(int(chr(key)))
    
    print (image + " Training Done.")

            
print "Training Completed."
print "Saving samples and responses to file."

#saving samples
np.savetxt('training_result\sudoku-samples.data', samples)

#making responses an array of float
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size,1))

#saving responses
np.savetxt('training_result\sudoku-responses.data', responses)
