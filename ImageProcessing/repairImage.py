# repair damaged images
import cv2
import matplotlib.pyplot as plt

damaged_image_path = "damaged_image.png"
damaged_image = cv2.imread(damaged_image_path, 0)
# cv2.imshow("img", damaged_image)
mask = cv2.imread(damaged_image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.inRange(mask, 0, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imshow("img", mask)
damaged_image = cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB)
output1 = cv2.inpaint(damaged_image, mask, 10, cv2.INPAINT_TELEA)
output2 = cv2.inpaint(damaged_image, mask, 10, cv2.INPAINT_NS)

img = [damaged_image, mask, output1, output2]
titles = ['damaged image', 'mask', 'TELEA', 'NS']

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(titles[i])
    plt.imshow(img[i])
plt.show()