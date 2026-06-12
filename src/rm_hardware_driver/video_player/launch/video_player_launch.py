import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('video_player')
    
    # Default config files
    params_file = os.path.join(pkg_dir, 'config', 'video_params.yaml')
    print(f"Using params file: {params_file}")
    video_path = 'package://video_player/video/output.avi'
    camera_info_url = 'package://video_player/config/camera_info.yaml'

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            name='video_path',
            default_value=video_path,
            description='Path to the video file to play'
        ),
        DeclareLaunchArgument(
            name='params_file',
            default_value=params_file,
            description='Path to the ROS2 parameters file'
        ),
        DeclareLaunchArgument(
            name='camera_info_url',
            default_value=camera_info_url,
            description='URL to the camera info file'
        ),
        DeclareLaunchArgument(
            name='use_sensor_data_qos',
            default_value='false',
            description='Use sensor data QoS profile'
        ),
        DeclareLaunchArgument(
            name='loop_playback',
            default_value='true',
            description='Loop video playback'
        ),
        DeclareLaunchArgument(
            name='fps',
            default_value='30.0',
            description='Playback frame rate'
        ),
        DeclareLaunchArgument(
            name='flip_image',
            default_value='false',
            description='Flip image both horizontally and vertically'
        ),
        
        # Video player node
        Node(
            package='video_player',
            executable='video_player_node',
            name='video_player',
            output='screen',
            emulate_tty=True,
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'video_path': LaunchConfiguration('video_path'),
                    'camera_info_url': LaunchConfiguration('camera_info_url'),
                    'use_sensor_data_qos': LaunchConfiguration('use_sensor_data_qos'),
                    'loop_playback': LaunchConfiguration('loop_playback'),
                    'fps': LaunchConfiguration('fps'),
                    'flip_image': LaunchConfiguration('flip_image'),
                },
            ],
        )
    ])
