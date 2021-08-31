from functions import *

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
#     cv2.imwrite("Frame.png", frame)
    edges = canny(frame)
#     cv2.imwrite("Edges.png", edges)
    cropped_canny = region_of_interest(edges)
#     cv2.imwrite("ROI.png", cropped_canny)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
#     cv2.imwrite("Lines.png", line_image)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
#     cv2.imwrite("Result.png", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
