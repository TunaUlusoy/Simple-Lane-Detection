from functions import *

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    edges = canny(frame)
    cropped_canny = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
