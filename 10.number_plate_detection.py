import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

img = cv2.imread("car0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(gray, 170, 200)
contours,_ = cv2.findContours(canny_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    contour_with_license_plate = None
    license_plate = None
    if area > 40:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)
        if len(approx) == 4:
            contour_with_license_plate = approx
            x,y,w,h = cv2.boundingRect(contour)
            license_plate = gray[y:y + h, x:x + w]
            break

license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
(thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

text = pytesseract.image_to_string(license_plate)
img = cv2.rectangle(img, (x,y),(x+w, y+h), (0,0,255), 3)
img = cv2.putText(img, text, (x-100, y-50), cv2.FONT_HERSHEY_COMPLEX, 3,(0,255,0), 6)

print("Licence plate: ", text)
cv2.imshow("License plate detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


