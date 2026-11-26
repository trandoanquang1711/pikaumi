# Thu Dual Pika Sense + RealSense D435i head RGB

Profile của máy này: D435i `317622076044`, firmware đọc được `5.15.1.55`, RGB 640×480 @ 30 Hz. Theo lựa chọn của người vận hành, depth và IMU head tắt. Các camera Pika giữ nguyên cấu hình và ảnh fisheye. Tính năng head nằm trong CLI hiện có, không cần cài thêm pyrealsense2 trên host hoặc chỉnh container bằng tay.

## Cấu hình và khởi chạy

Trong `config/sensors.yaml`:

```yaml
head_camera:
  enabled: true
  serial: "317622076044"
  enable_depth: false
```

Khi chuyển máy/camera, dùng serial do `rs-enumerate-devices -s` trong container báo. Không dùng USB descriptor serial. Head serial phải khác hai D405 cổ tay. Có thể đặt `enabled: false` để thu dataset hai camera như trước.

Từ workspace root, sau khi container `pika-ros` chạy và đã hiệu chuẩn Vive:

```bash
export DATA_DIR="$HOME/Data/dual_pika_head"
export PIKA_VLA_PYTHON="$PWD/.venv-pika-dataset/bin/python"
./pika_data.sh sensors --camera-fps 30 --use-head
```

Giữ terminal này mở; cả Pika và head dùng chung một roslaunch và khóa chống chạy trùng. Dừng bằng Ctrl+C và chờ thoát trước khi đổi cấu hình. Trong terminal khác, đặt lại biến môi trường (biến export trong terminal A không tự có ở terminal B). Ổ phụ của máy hiện tại mount tại `/media/ittest/recdata1`:

```bash
export DATA_DIR="/media/ittest/recdata1/dual_pika_head"
export PIKA_VLA_PYTHON="$PWD/.venv-pika-dataset/bin/python"
./pika_data.sh status --check-streams --use-head --fps 30 --data-root "${DATA_DIR:?Dat DATA_DIR trong terminal nay}/raw"
./pika_data.sh session --use-head \
  --instruction "fold a towel" \
  --data-root "${DATA_DIR:?Dat DATA_DIR trong terminal nay}/raw" \
  --output-root "${DATA_DIR:?Dat DATA_DIR trong terminal nay}/processed" \
  --calibration-dir "$PWD/data/pika_calibration" \
  --fps 30 --repo-id local/dual_pika_head
```

GUI kiểm tra lại streams trước mỗi lần Start. Save ghi cả ba RGB, trạng thái tracking, instruction và `capture_metadata.json`; thiếu ảnh hoặc CameraInfo head sẽ báo lỗi. Dùng Discard cho lần thu lỗi. Convert session chạy luồng export → prepare → LeRobot hiện có. CLI `record N "instruction" --use-head` cũng thu head; chạy `status --check-streams` trước khi dùng CLI.

Phải dùng session/output root mới cho dataset ba camera; không tái sử dụng cache labels cũ. Không trộn episode hai camera và ba camera vào cùng một LeRobot dataset. Export từ raw mới tự nhận diện head qua metadata; `--vista-only` giờ nghĩa là bỏ D405 cổ tay, vẫn giữ head nếu episode có head.

## Chọn head theo từng task

- Session mới/CLI record: `--use-head` thu cả head; `--no-use-head` hoặc bỏ cờ chỉ thu hai camera Pika.
- `session.json` lưu `settings.use_head`. Khi mở lại session không truyền cờ, hệ thống dùng lựa chọn đã lưu; cờ trái với session cũ sẽ báo lỗi, cần tạo session mới cho task khác. Session cũ chưa có trường này được nhận diện từ metadata/raw đã thu.
- Head có thể tiếp tục publish ở terminal sensors khi task không dùng head; recorder và preflight task đó không subscribe/yêu cầu/lưu head. Để tắt hẳn driver, dừng terminal sensors rồi chạy `sensors --no-use-head`. Khi cần bật lại, dùng `sensors --use-head`.
- `status --check-streams --no-use-head` chỉ kiểm tra Pika; `--use-head` kiểm tra cả head. Bỏ cờ ở `sensors`/`status` giữ hành vi theo profile cảm biến.
- `--use-head` khi head chưa được khởi động sẽ báo lỗi kèm lệnh cần chạy. Lựa chọn capture dùng môi trường riêng của tiến trình, không sửa YAML hoặc cấu hình ROS chung.

Ví dụ đổi task trên cùng sensor stack:

```bash
./pika_data.sh session --use-head --instruction "pick and place" \
  --data-root "${DATA_DIR:?}/raw" --output-root "${DATA_DIR:?}/processed" --fps 30
./pika_data.sh session --no-use-head --instruction "fold a towel" \
  --data-root "${DATA_DIR:?}/raw" --output-root "${DATA_DIR:?}/processed" --fps 30
```

Mỗi lần mở không có `--session-dir` sẽ tạo session riêng. Các episode giữ metadata camera riêng, nên conversion tự nhận đúng số view từ raw.

## Đồng bộ và hợp đồng dữ liệu

| Thành phần | Quy tắc |
|---|---|
| Nguồn thời gian | `Image.header.stamp` ROS, tên file raw được đọc bằng Decimal để giữ độ chính xác epoch |
| Pose/gripper | Nội suy theo grid dataset; gap tối đa mặc định 30 ms |
| Wrist pair | Ghép causal, tuổi ảnh tối đa 40 ms; wrist skew mặc định 10 ms ở `--vista-only` |
| Head | Chọn frame gần thời điểm wrist pair nhất nhưng không muộn hơn tick; tuổi ảnh tối đa `--max-image-offset-ms` (mặc định 40 ms) |
| Skew ba RGB | Tối đa 40 ms giữa timestamp nhỏ nhất và lớn nhất; độc lập ngưỡng wrist 10 ms |
| Mất stream | Preflight từ chối; export/prepare loại tick và cắt segment, không nối action qua khoảng mất ảnh |
| RGB | BGR raw → RGB HDF5, native aspect ratio; mặc định không resize |
| Labels | State 11D và action 14D TCP-local giữ nguyên |

Đây là đồng bộ phần mềm theo timestamp, chưa hiệu chuẩn latency phơi sáng camera so với pose, và không có dây trigger chung giữa D435i và Pika fisheye. `enable_sync` của RealSense không đồng bộ phần cứng các camera Pika. Không diễn giải ngưỡng 40 ms thành sai số phơi sáng đã được chứng nhận. Xem [ROS 1 RealSense wrapper](https://github.com/realsenseai/realsense-ros/tree/ros1-legacy) để đối chiếu ý nghĩa tham số driver.

Preflight kiểm tra stream thiếu, timestamp trùng/đảo, clock lệch/stale, ảnh rỗng và tỷ lệ tick ghép hợp lệ (ít nhất 90%, có đoạn liên tục ≥1 giây). Không nới ngưỡng để che lỗi USB/CPU. Gắn D435i chắc chắn, ưu tiên USB 3 riêng; kiểm tra toàn bộ hệ thống cùng chạy vì phép đo camera đơn không xác nhận băng thông tổng.

## Nội dung bàn giao

```text
raw/episodeN/
  camera/color/headCamera/{<timestamp>.jpg,config.json}
  capture_metadata.json
processed/<session>/
  source/episodeN/pika_source_30hz.hdf5
  prepared/episodeN/{labels.npz,report.json}
  lerobot/
    data/..., meta/...
    videos/observation.images.left_wrist/...
    videos/observation.images.right_wrist/...
    videos/observation.images.head/...
    provenance/episodeN_timestamps.npz
    provenance/episodeN_calibration.json
    pika_source_reports.json
```

Raw codec/suffix do recorder Pika quyết định; exporter nhận cả JPEG và PNG. HDF5 `source_episodes/episodeN/images/head_rgb` chứa RGB uint8, source_files, timestamp_ns, offset_ns, K, D và camera_model. CameraInfo head phải có tiêu cự dương, hữu hạn, đúng kích thước ảnh. Không tự tạo head-to-world extrinsic; transform này chưa được đo. Không resize ảnh sang tỷ lệ khác khi đặt `--image-size`.

LeRobot chứa ba feature `observation.images.{left_wrist,right_wrist,head}`. Video được mã hóa theo cơ chế LeRobot; timestamp gốc không được lấy từ PTS của MP4. Sidecar `provenance/*_timestamps.npz` lưu đúng thứ tự hàng label của từng raw episode: `source_row`, `timestamp_ns`, `{left_wrist,right_wrist,head}_timestamp_ns`. Kết hợp `segments` trong `pika_source_reports.json` để ánh xạ sang episode LeRobot. File calibration lưu K/D theo kích thước ảnh xuất và metadata nguồn. Các đường dẫn `provenance` trong báo cáo là tương đối, có thể di chuyển cùng dataset.

Copy toàn bộ `lerobot/` lên máy GPU, bao gồm `provenance/`. Giữ raw/HDF5 riêng để audit hoặc xuất lại. Chia train/validation theo raw episode. Bộ kiểm tra `pika_pi05.py` nhận cả hai hoặc ba camera và đưa head vào input features; khi dùng dataset ba view với model khác, phải cấu hình model để thực sự đọc `observation.images.head`. Bridge điều khiển robot hiện hữu cần được tích hợp camera head riêng trước khi triển khai policy ba view lên robot.

Có thể tạo checksum trong thư mục `lerobot/` trước khi nén/bàn giao:

```bash
find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
sha256sum -c checksums.sha256
```

Depth là tính năng tùy chọn cho lần thu khác: `enable_depth: true` lưu aligned uint16 PNG/HDF5, đơn vị 0.001 m theo driver ROS hiện cài. Depth không được đưa vào MP4 RGB hoặc feature huấn luyện head. Profile hiện tại chỉ thu RGB.

## Kết quả kiểm chứng và bước nghiệm thu còn lại

Đã kiểm tra trực tiếp D435i: 120 frame/4 giây, 30.004 Hz, 640×480, timestamp tăng liên tục, receive age lớn nhất 35.49 ms trong phép đo. K đọc được: fx=608.146, fy=608.182, cx=322.969, cy=247.383. Các giá trị này là kết quả kiểm tra; conversion luôn lấy CameraInfo được ghi trong episode. Launch tổng hợp Pika + head đã resolve bằng `roslaunch --dump-params` trên container thực tế. Camera kiểm tra đã được dừng.

Các regression offline kiểm tra ghép causal, camera thiếu/sai clock/dropout, round-trip RGB/depth/intrinsics, labels loại head tương lai và sidecar timestamp/calibration. Trước khi phát hành dataset thật cần:

1. Chạy cả Pika và head rồi đạt `status --check-streams` ở fps sẽ thu.
2. Thu một demonstration đủ dài, Save, Convert và đọc lại dataset bằng LeRobot trong môi trường có đủ dependencies trong `requirements-vla-cpu.txt`.
3. Kiểm tra video ba góc nhìn của cùng thao tác, báo cáo tracking/segment, timestamp sidecar, và checksum sau khi copy.

Đã cài môi trường CPU riêng tại `.venv-pika-dataset` (bộ kiểm thử ban đầu chạy trong `/tmp/pika-head-validation`) theo các phiên bản trong `requirements-vla-cpu.txt`, cùng torch 2.7.1+cpu/torchvision 0.22.1+cpu, không thay đổi `.venv-pika` hiện hữu. Kiểm thử `test_pika_head_lerobot.py` đã mã hóa MP4 thật, đọc lại đủ ba camera/state/action, di chuyển dataset rồi ẩn HDF5 nguồn để xác nhận tính độc lập, kiểm tra provenance và từ chối dataset trộn hai/ba view. Fine-tuning Pi0.5 trên GPU và một demonstration Pika + head thực tế chưa được nghiệm thu.

Để chạy lại bài kiểm thử đóng gói trong môi trường đã có dependencies dataset và pytest:

```bash
PYTHONPATH="$PWD/umi-vista-main/post_training/lerobot/src" \
  python -m pytest Pika_Data_source_2026-09-30/Pika_Data/tests/test_pika_head_lerobot.py -q
```

`.venv-pika` hiện hữu trên máy vẫn được giữ nguyên. Dùng `export PIKA_VLA_PYTHON="$PWD/.venv-pika-dataset/bin/python"` trước khi mở session hoặc chạy prepare/lerobot. Toàn bộ 195 test offline đã đạt trong môi trường CPU đủ dependencies, gồm kiểm thử LeRobot thật; dry-run bridge 10 bước đã đạt với hardware_tx_count=0. Không coi phép đo camera đơn hay dataset synthetic là nghiệm thu đồng bộ toàn bộ thiết bị.

Khi đóng gói phần mềm sang máy mới, không copy venv. Tạo lại bằng các lệnh sau (sau đó đặt `PIKA_VLA_PYTHON` như trên):

```bash
uv venv --python 3.12 .venv-pika-dataset
uv pip install --python .venv-pika-dataset/bin/python \
  --index https://download.pytorch.org/whl/cpu torch==2.7.1+cpu torchvision==0.22.1+cpu
uv pip install --python .venv-pika-dataset/bin/python \
  -r Pika_Data_source_2026-09-30/Pika_Data/data_tools/requirements-vla-cpu.txt \
  pytest pyyaml rich
```

