import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from skimage.morphology import skeletonize

# in:
# out:
# desc:
def get8BitFitsData(data):
		# Clip data to brightness limits
	a = 150
	b = 500

	data[data > b] = b
	data[data < a] = a
			
		# Scale data to range [0, 1] 
	data = (data - a)/(b - a)
			
		# Convert to 8-bit integer  
	data = (255*data).astype(np.uint8)
			
		# Invert y axis
	data = data[::-1, :]

	return data

# in:
# out:
# desc:
def showImage(data):
	# Create image from data array
	image = Image.fromarray(data, 'L')

	plt.imshow(image)
	plt.gray()
	plt.show()


# in: 
# out:
# desc:
def openImage(imPath):
	# Try to read data from first HDU in fits file
	data = fits.open(imPath)[0].data
				# If nothing is there try the second one
	if data is None:
		data = fits.open(imPath)[1].data

	data = get8BitFitsData(data)

	img_cv = cv2.resize(data,(data.shape[1],data.shape[0]))

	return img_cv


def normalize(img):
	imgS = img.copy() 
	dst = np.zeros(shape=(5,5))
	imgNorm=cv2.normalize(imgS,dst,0,255,cv2.NORM_MINMAX)
	return imgNorm

# in: 
# out:
# desc:
# gaussian blur
def gaussBlur(img):
	kernel_size = 7
	blurGrayImage = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

	return blurGrayImage


# in: 
# out:
# desc:
#binarizare
def binImage(img):
	retval, binaryImage = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
	return binaryImage


# in: 
# out:
# desc:
def lineDetection(img):
	# line detect
	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 180  # angular resolution in radians of the Hough grid
	threshold = 8  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 8  # minimum number of pixels making up a line
	max_line_gap = 3  # maximum gap in pixels between connectable line segments

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
	                    min_line_length, max_line_gap)

	return lines


# in: 
# out:
# desc:
def starsDetection(data):
	#hdu = datasets.load_star_image()    
	mean, median, std = sigma_clipped_stats(data, sigma=3.0)    

	daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)    
	sources = daofind(data - median)    
	for col in sources.colnames:    
		sources[col].info.format = '%.8g'  # for consistent table output

	return sources


# in: 
# out:
# desc:
def starErase(img):
	sources = starsDetection(img)

	imgS = np.copy(img)
	for i in range(len(sources)):
		xCen = sources[i]['xcentroid']
		yCen = sources[i]['ycentroid']

		cv2.circle(imgS,(int(round(xCen)),int(round(yCen))), 3, (0,0,0), -1)

	return imgS


# in: 
# out:
# desc:
def plotLineDetection(img,lines):
	maxLenPos = 0
	maxLen = 0

	for i in range(0,len(lines)):
		le = np.sqrt((lines[i][0][0]-lines[i][0][2])**2 + (lines[i][0][1]-lines[i][0][3])**2)
		if le > maxLen:
			maxLenPos = i
			maxLen = le
	imgS = np.copy(img)
	cv2.line(imgS,(lines[maxLenPos][0][0],lines[maxLenPos][0][1]),(lines[maxLenPos][0][2],lines[maxLenPos][0][3]),(255,255,255),2)
	return imgS



def skeletonize(img):
    
	img = img.copy() 
	skel = img.copy()

	skel[:,:] = 0
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

	while True:
		eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
		temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
		temp  = cv2.subtract(img, temp)
		skel = cv2.bitwise_or(skel, temp)
		img[:,:] = eroded[:,:]
		if cv2.countNonZero(img) == 0:
			break

	return skel



img = openImage("Galileo103B023.FIT")

imgNorm = normalize(img)

noStarsImg = starErase(imgNorm)

noStarsBlurImg = gaussBlur(noStarsImg)
#edges = cv2.Canny(noStarsImg, 75, 150)

binWithSatelliteImg = binImage(noStarsBlurImg)
#skeleton = skeletonize(binWithSatelliteImg)

lines = lineDetection(binWithSatelliteImg)

if lines is not None:
	print(lines)
	imgL = plotLineDetection(img, lines)



f, axarr = plt.subplots(3,2)
plt.gray()

titles = ["Original", "Normalize", "WithoutStars","Blur","Binary","DetectionSat"]
images = [img,imgNorm,noStarsImg,noStarsBlurImg,binWithSatelliteImg,imgL]

for i in range(len(images)):
	plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
	plt.title(titles[i])


plt.show()




cv2.waitKey(0)
cv2.destroyAllWindows()

