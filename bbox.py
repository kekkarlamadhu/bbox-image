import cv2
import numpy as np

# Load image and remove watermark
image = cv2.imread('image.png')

alpha = 2.0
beta = -160
new = alpha * image + beta
new = np.clip(new, 0, 255).astype(np.uint8)
cv2.imwrite("cleaned.png",new)
image2=cv2.imread("cleaned.png")




# Grayscale, Gaussian blur, Otsu's threshold of Image
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

blur = cv2.bilateralFilter(gray, 9,75,75)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
dilate = cv2.dilate(thresh, kernel, iterations=6)

# Find contours and draw rectangle
contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if cv2.contourArea(c)<3000:
        continue
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)


cv2.imshow('image', image)
cv2.imwrite('output.png',image)
cv2.waitKey()
