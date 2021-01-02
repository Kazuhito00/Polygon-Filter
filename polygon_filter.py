import copy
import numpy as np
import cv2 as cv


def polygon_filter(image,
                   akaze_threshold=0.00001,
                   additional_points=[],
                   draw_line=False):
    """ポリゴンフィルターを適用した画像を返す

    Args:
        image: OpenCV Image
        akaze_threshold: AKAZE Threshold
        additional_point: Subdiv2D Points for additional Insert
        draw_line: Whether to draw the sides of the triangle

    Returns:
        Image after applying the filter.
    """
    height, width, _ = image.shape[0], image.shape[1], image.shape[2]

    # 特徴点抽出
    akaze = cv.AKAZE_create(threshold=akaze_threshold)
    key_points, _ = akaze.detectAndCompute(image, None)
    key_points = cv.KeyPoint_convert(key_points)

    # ドロネー図作成
    subdiv = cv.Subdiv2D((0, 0, width, height))

    subdiv.insert((0, 0))
    subdiv.insert((width - 1, 0))
    subdiv.insert((0, height - 1))
    subdiv.insert((width - 1, height - 1))
    subdiv.insert((int(width / 2), 0))
    subdiv.insert((0, int(height / 2)))
    subdiv.insert((width - 1, int(height / 2)))
    subdiv.insert((int(width / 2), height - 1))
    subdiv.insert((int(width / 2), int(height / 2)))
    for key_point in key_points:
        subdiv.insert((int(key_point[0]), int(key_point[1])))
    for additional_point in additional_points:
        subdiv.insert((int(additional_point[0]), int(additional_point[1])))

    triangle_list = subdiv.getTriangleList()
    triangle_polygons = triangle_list.reshape(-1, 3, 2)

    # ドロネー三角形用の色取得
    triangle_info_list = []
    for triangle_polygon in triangle_polygons:
        pt1 = (int(triangle_polygon[0][0]), int(triangle_polygon[0][1]))
        pt2 = (int(triangle_polygon[1][0]), int(triangle_polygon[1][1]))
        pt3 = (int(triangle_polygon[2][0]), int(triangle_polygon[2][1]))
        pt0 = (
            int((pt1[0] + pt2[0] + pt3[0]) / 3),
            int((pt1[1] + pt2[1] + pt3[1]) / 3),
        )
        color = tuple(image[pt0[1], pt0[0]])
        color = (int(color[0]), int(color[1]), int(color[2]))

        triangle_info_list.append([pt1, pt2, pt3, color])

    # 描画
    for triangle_info in triangle_info_list:
        pt1 = (int(triangle_info[0][0]), int(triangle_info[0][1]))
        pt2 = (int(triangle_info[1][0]), int(triangle_info[1][1]))
        pt3 = (int(triangle_info[2][0]), int(triangle_info[2][1]))
        contours = np.array([
            [pt1[0], pt1[1]],
            [pt2[0], pt2[1]],
            [pt3[0], pt3[1]],
        ])

        cv.fillConvexPoly(image, points=contours, color=triangle_info[3])

        if draw_line:
            cv.line(image, pt1, pt2, (255, 255, 255), 1, 8, 0)
            cv.line(image, pt2, pt3, (255, 255, 255), 1, 8, 0)
            cv.line(image, pt3, pt1, (255, 255, 255), 1, 8, 0)

    return image


def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.resize(frame, (960, 540))
        original_frame = copy.deepcopy(frame)

        frame = polygon_filter(frame,
                               akaze_threshold=0.0002,
                               additional_points=[[100, 0], [200, 0]],
                               draw_line=True)

        cv.imshow('original', original_frame)
        cv.imshow('polygon filter', frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()