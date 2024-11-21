import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8モデルの読み込み
model = YOLO('sample_model.pt')

# 動画の読み込み
cap = cv2.VideoCapture('../videos/sample.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)  # 映像のフレームレート

writer = cv2.VideoWriter('../videos/results.mp4', fourcc=fourcc, fps=fps, frameSize=(width, height))

# 最初のフレームを取得
ret, frame1 = cap.read()
if not ret:
    print("動画の読み込みに失敗しました")
    exit()

# 物体検出関数
def detect_objects(frame):
    detection_targets = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
    results = model(frame)  # YOLOで検出
    points = []
    confidences = []
    class_ids = []

    boxes = results[0].boxes

    # 検出された物体を処理
    for box in boxes:
        cls = box.cls
        conf = box.conf.cpu()
        conf = conf.item()
        xyxy = box.xyxy[0].cpu()

        target_name = model.names[int(cls)]

        if target_name in detection_targets and conf > 0.5:  # 信頼度が0.5より大きい場合に選択
            points.append(xyxy)
            confidences.append(conf)
            class_ids.append(int(cls))
    
    return points, confidences, class_ids

# 最初のフレームをグレースケールに変換
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# 物体ごとに特徴点を保持
prev_pts_dict = {}

# 物体の移動を追跡するための変数
prev_positions = {}

frame_num = 1

# フレームごとに物体の追跡と速度推定
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    # YOLOで物体検出
    points, confidences, class_ids = detect_objects(frame2)

    # 物体の位置を描画
    for i, box in enumerate(points):
        x1, y1, x2, y2 = box
        label = f'{model.names[class_ids[i]]}'
        confidence = confidences[i]
        cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame2, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 各物体の特徴点を保持
        prev_pts_dict[i] = np.array([[x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]], dtype=np.float32)

        # 初回の位置を保存
        if i not in prev_positions:
            prev_positions[i] = (x1 + x2) / 2, (y1 + y2) / 2

    # オプティカルフローを使用して物体の動きを追跡
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    for i, box in enumerate(points):
        prev_pts = prev_pts_dict[i]

        # 物体が移動した距離を計算
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

        if status[0] == 1:
            a, b = next_pts[0].ravel()
            c, d = prev_pts[0].ravel()

            # 物体の移動距離を計算
            distance = np.sqrt((a - c) ** 2 + (b - d) ** 2)
            speed = distance * fps  # ピクセル/秒の速度に変換

            # 遠近感補正（物体のy座標を使って補正）
            scale_factor = 1 / (1 + (b / frame2.shape[0]))  # y座標を基にスケールを調整
            adjusted_speed = speed * scale_factor

            # スケール調整（ピクセル/秒 → m/s → km/h）
            speed_m_per_s = adjusted_speed * 0.05
            speed_km_per_h = speed_m_per_s * 3.6

            # 速度を物体のボトム下に表示
            x1, y1, x2, y2 = points[i]
            cv2.putText(frame2, f'Speed: {speed_km_per_h:.2f} km/h', (int(x1), int(y2) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 現在の位置を更新
            prev_positions[i] = a, b

    # 次のフレームの処理のために更新
    prev_gray = gray.copy()

    # 結果を表示
    cv2.imshow('Frame', frame2)
    cv2.imwrite(f'../images/frame_{frame_num:04d}.png', frame2)

    frame_num += 1

    writer.write(frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
