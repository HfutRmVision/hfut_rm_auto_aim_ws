# Video Player

一个 ROS2 虚拟摄像头节点，从视频文件中读取图像帧并发布到 `/image_raw` 话题，便于调试和开发。

## 功能特性

- 从视频文件读取并发布图像帧到 ROS2 话题
- 支持所有 OpenCV 支持的视频格式 (MP4, AVI, MOV 等)
- 可配置播放帧率
- 支持循环播放
- 支持图像翻转
- 发布相机信息 (camera_info)
- 类似 `ros2_mindvision_camera` 的接口

## 依赖项

- ROS2 (Humble/Iron/Rolling)
- OpenCV
- cv_bridge
- image_transport
- camera_info_manager

## 编译

```bash
cd ~/hfut_rm_auto_aim_ws
colcon build --packages-select video_player
source install/setup.bash
```

## 使用方法

### 基本用法

使用 launch 文件启动，需要指定视频路径：

```bash
ros2 launch video_player video_player_launch.py video_path:=/path/to/your/video.mp4
```

### 高级用法

可以通过 launch 参数自定义各种选项：

```bash
ros2 launch video_player video_player_launch.py \
  video_path:=/path/to/your/video.mp4 \
  fps:=60.0 \
  loop_playback:=true \
  flip_image:=false \
  camera_info_url:=package://video_player/config/camera_info.yaml
```

### 直接运行节点

也可以直接运行节点并通过参数设置：

```bash
ros2 run video_player video_player_node \
  --ros-args \
  -p video_path:=/path/to/your/video.mp4 \
  -p fps:=30.0 \
  -p loop_playback:=true
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `video_path` | string | "" | 视频文件路径（必需） |
| `fps` | double | 30.0 | 播放帧率 |
| `loop_playback` | bool | true | 是否循环播放 |
| `camera_name` | string | "camera" | 相机名称 |
| `frame_id` | string | "camera_optical_frame" | 图像和 CameraInfo 的 frame_id |
| `flip_image` | bool | false | 是否翻转图像（水平+垂直） |
| `use_sensor_data_qos` | bool | false | 是否使用 sensor_data QoS |
| `camera_info_url` | string | "" | 相机标定文件 URL (可选) |

## 发布的话题

- `/image_raw` (sensor_msgs/Image): 图像数据
- `/camera_info` (sensor_msgs/CameraInfo): 相机标定信息

## 配置文件

### video_params.yaml

包含所有可配置参数的示例文件。

### camera_info.yaml

相机标定信息，包括相机内参、畸变系数等。可以根据实际相机标定结果修改。

## 示例

### 调试目标检测算法

```bash
# 使用录制的视频进行目标检测测试
ros2 launch video_player video_player_launch.py \
  video_path:=~/Videos/test_recording.mp4 \
  fps:=30.0
```

### 在没有实际摄像头的环境下开发

```bash
# 使用预先录制的视频代替实际摄像头
ros2 launch video_player video_player_launch.py \
  video_path:=~/Videos/camera_simulation.mp4 \
  loop_playback:=true
```

## 注意事项

1. 确保 `video_path` 参数正确指向一个有效的视频文件
2. 如果视频帧率与设置的 fps 不匹配，节点会按照设置的 fps 播放
3. 循环播放模式下，视频结束后会自动重新开始
4. 可以使用 `rqt_image_view` 查看发布的图像：
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```

## 故障排除

### 视频无法打开
- 检查视频文件路径是否正确
- 确认视频文件格式被 OpenCV 支持
- 检查文件权限

### 图像无法显示
- 使用 `ros2 topic list` 确认话题是否发布
- 使用 `ros2 topic echo /image_raw` 查看是否有数据
- 检查 QoS 设置是否匹配

## License

MIT License
