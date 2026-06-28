import cv2
import easyocr

# Load the image
image = cv2.imread('./images/image_3.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to highlight the text
_, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Run EasyOCR on the preprocessed image
result = reader.readtext(threshold_image)
print(f"result  : {result}")

# Extract numbers from the result
numbers = [text[1] for text in result if text[1].isdigit()]

print(f"Detected numbers: {numbers}")

