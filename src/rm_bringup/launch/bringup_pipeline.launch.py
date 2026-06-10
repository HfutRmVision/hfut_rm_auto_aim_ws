#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GimbalPipeline 统一节点启动文件

数据流 (节点内部零延迟):
  装甲板检测 ──→ [ gimbal_pipeline ] ──→ cmd_gimbal
                       │   内部: tracker → selector → controller
                       │
                  ballistic_solver (外部服务)

相比 bringup_max_entropy_test.launch.py 的区别:
  - 原来 max_entropy_tracker + target_selector + gimbal_controller 三个独立节点
    合并为单个 gimbal_pipeline 节点
  - 消除了两跳 ROS2 话题通信延迟 (tracked_robots, selected_target)
  - 其余节点 (camera, detector, serial, ballistic_solver) 保持不变

启动方式:
    ros2 launch rm_bringup bringup_pipeline.launch.py
    ros2 launch rm_bringup bringup_pipeline.launch.py image_source:=video virtual_serial:=true debug:=true
"""

import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition

sys.path.append(os.path.join(get_package_share_directory('rm_bringup'), 'launch'))


def generate_launch_description():
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node, PushRosNamespace
    from launch.actions import TimerAction, DeclareLaunchArgument, OpaqueFunction
    from launch.substitutions import LaunchConfiguration
    from launch import LaunchDescription

    # ── 加载启动参数配置 ──
    try:
        launch_params = yaml.safe_load(open(os.path.join(
            get_package_share_directory('rm_bringup'), 'config', 'launch_params_decoupled.yaml')))
    except Exception:
        launch_params = {
            'robot': 'default',
            'image_source': 'video',
            'virtual_serial': True,
            'namespace': '',
            'odom2camera': {
                'xyz': '0 0 0',
                'rpy': '0 0 0'
            }
        }

    robot_name = str(launch_params.get('robot', 'default')).strip() or 'default'
    bringup_config_root = os.path.join(get_package_share_directory('rm_bringup'), 'config')

    # ── 声明启动参数 ──
    declare_image_source = DeclareLaunchArgument(
        'image_source',
        default_value=str(launch_params.get('image_source', 'video')),
        description='Image source: video | mindvision | hik'
    )
    declare_virtual_serial = DeclareLaunchArgument(
        'virtual_serial',
        default_value=str(launch_params.get('virtual_serial', True)).lower(),
        description='Use virtual serial instead of real serial'
    )
    declare_debug = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Enable debug mode for all nodes'
    )
    declare_namespace = DeclareLaunchArgument(
        'namespace',
        default_value=launch_params.get('namespace', ''),
        description='Namespace for all nodes'
    )
    declare_detector_type = DeclareLaunchArgument(
        'detector_type',
        default_value=str(launch_params.get('detector_type', 'armor_detector')),
        description='Detector type: armor_detector | armor_detector_nn'
    )
    declare_use_container = DeclareLaunchArgument(
        'use_camera_detector_container',
        default_value=str(launch_params.get('use_camera_detector_container', True)).lower(),
        description='Put camera and detector in same container'
    )
    # declare_enable_auto_buff = DeclareLaunchArgument(
    #     'enable_auto_buff',
    #     default_value=str(launch_params.get('enable_auto_buff', False)).lower(),
    #     description='Enable auto_buff detector + pose_estimator pipeline'
    # )

    # ── URDF 机器人描述 ──
    robot_gimbal_description = Command(['xacro ', os.path.join(
        get_package_share_directory('rm_robot_description'), 'urdf', 'rm_gimbal.urdf.xacro'),
        ' xyz:=', launch_params['odom2camera']['xyz'],
        ' rpy:=', launch_params['odom2camera']['rpy']])

    robot_gimbal_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(robot_gimbal_description, value_type=str),
            'publish_frequency': 1000.0
        }]
    )

    def get_bringup_params(name):
        # Use flat robot-specific config only when a concrete robot (not 'default') is selected
        if robot_name and robot_name != 'default':
            robot_specific = os.path.join(bringup_config_root, robot_name, f"{name}_params.yaml")
            if os.path.isfile(robot_specific):
                return robot_specific
        # fallback to shared node_params
        return os.path.join(bringup_config_root, 'node_params', f"{name}_params.yaml")

    def get_pkg_params(pkg_name, param_file):
        return os.path.join(get_package_share_directory(pkg_name), 'config', param_file)

    def get_pkg_params_with_robot_override(pkg_name, param_file):
        # package default always first; if robot-specific flat override exists, append it
        default_path = get_pkg_params(pkg_name, param_file)
        if robot_name and robot_name != 'default':
            override_path = os.path.join(bringup_config_root, robot_name, param_file)
            print(f"Checking for robot-specific override params at: {override_path}")
            if os.path.isfile(override_path):
                return [default_path, override_path]
        return [default_path]

    # ==================== 装甲板检测节点 (ComposableNode) ====================
    detector_type = str(launch_params.get('detector_type', 'armor_detector'))
    print(f"Selected detector type: {detector_type}")

    if detector_type == 'armor_detector_nn':
        armor_detector_node = ComposableNode(
            package='armor_detector_nn',
            plugin='fyt::auto_aim::ArmorDetectorNNNode',
            name='armor_detector',
            parameters=[get_pkg_params('armor_detector_nn', 'armor_detector_nn.yaml')],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    else:
        armor_detector_node = ComposableNode(
            package='armor_detector',
            plugin='fyt::auto_aim::ArmorDetectorNode',
            name='armor_detector',
            parameters=[get_bringup_params('armor_detector')],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    # ==================== 弹道解算服务 ====================
    ballistic_solver_node = Node(
        package='ballistic_solver',
        executable='ballistic_solver_node_exe',
        name='ballistic_solver',
        output='screen',
        emulate_tty=True,
        parameters=[get_pkg_params('ballistic_solver', 'ballistic_solver.yaml')],
    )

    # ==================== GimbalPipeline 统一节点 ====================
    # 替代原来的 max_entropy_tracker + target_selector + gimbal_controller
    gimbal_pipeline_node = Node(
        package='gimbal_pipeline',
        executable='gimbal_pipeline_node',
        name='gimbal_pipeline',
        output='both',
        emulate_tty=True,
        parameters=[
            *get_pkg_params_with_robot_override('gimbal_pipeline', 'gimbal_pipeline.yaml'),
            {
                'debug_mode': LaunchConfiguration('debug')
            }
        ],
        remappings=[
            # 输出 cmd_gimbal 保持兼容原 serial_driver 订阅的话题名
            ('cmd_gimbal', '/armor_solver/cmd_gimbal'),
        ],
    )

    # ==================== AutoBuff 链路 (可选) ====================
    # buff_detector_node = Node(
    #     package='auto_buff',
    #     executable='buff_detector_node',
    #     name='buff_detector',
    #     output='both',
    #     emulate_tty=True,
    #     parameters=get_pkg_params_with_robot_override('auto_buff', 'buff_detector.yaml'),
    #     condition=IfCondition(LaunchConfiguration('enable_auto_buff')),
    # )

    # buff_pose_estimator_node = Node(
    #     package='auto_buff',
    #     executable='buff_pose_estimator_node',
    #     name='buff_pose_estimator',
    #     output='both',
    #     emulate_tty=True,
    #     parameters=get_pkg_params_with_robot_override('auto_buff', 'buff_pose_estimator.yaml'),
    #     condition=IfCondition(LaunchConfiguration('enable_auto_buff')),
    # )

    # ==================== 相机+检测器 容器 ====================
    use_container = str(launch_params.get('use_camera_detector_container', True)).lower() == 'true'

    # 根据 image_source 获取相机节点的 (package, executable, parameters) 配置
    # def get_camera_node_config(image_source):
    #     if image_source == 'video':
    #         return {
    #             'package': 'video_player',
    #             'executable': 'video_player_node',
    #             'name': 'video_player',
    #             'parameters': [get_bringup_params('video_player')],
    #         }
    #     elif image_source == 'mindvision':
    #         return {
    #             'package': 'mindvision_camera',
    #             'executable': 'mindvision_camera_node',
    #             'name': 'mv_camera',
    #             'parameters': [get_bringup_params('mindvision_camera_driver')],
    #         }
    #     elif image_source == 'hik':
    #         return {
    #             'package': 'ros2_hik_camera',
    #             'executable': 'ros2_hik_camera_node',
    #             'name': 'hik_camera',
    #             'parameters': [get_bringup_params('hik_camera_driver')],
    #             'env': {'MVCAM_SDK_PATH': '/opt/MVS',
    #                      'MVCAM_COMMON_RUNENV': '/opt/MVS/lib'},
    #         }
    #     else:
    #         return {
    #             'package': 'mindvision_camera',
    #             'executable': 'mindvision_camera_node',
    #             'name': 'camera_driver',
    #             'parameters': [get_bringup_params('camera_driver')],
    #         }

    # 获取 detector 独立节点配置
    def get_detector_node_config():
        if detector_type == 'armor_detector_nn':
            return {
                'package': 'armor_detector_nn',
                'executable': 'armor_detector_nn_node',
                'name': 'armor_detector',
                'parameters': [get_pkg_params('armor_detector_nn', 'armor_detector_nn.yaml')],
            }
        else:
            return {
                'package': 'armor_detector',
                'executable': 'armor_detector_node',
                'name': 'armor_detector',
                'parameters': [get_bringup_params('armor_detector')],
            }

    # Build the ComposableNode for container mode (reuses original logic)
    def _make_camera_composable_node(image_source):
        if image_source == 'video':
            return ComposableNode(
                package='video_player',
                plugin='video_player::VideoPlayerNode',
                name='video_player',
                parameters=[get_bringup_params('video_player')],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        elif image_source == 'mindvision':
            return ComposableNode(
                package='mindvision_camera',
                plugin='mindvision_camera::MVCameraNode',
                name='mv_camera',
                parameters=[get_bringup_params('mindvision_camera_driver')],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        elif image_source == 'hik':
            return ComposableNode(
                package='ros2_hik_camera',
                plugin='ros2_hik_camera::HikCameraNode',
                name='hik_camera',
                parameters=[get_bringup_params('hik_camera_driver')],
                extra_arguments=[{'use_intra_process_comms': True,
                                  'env': {
                                    'MVCAM_SDK_PATH': '/opt/MVS',
                                    'MVCAM_COMMON_RUNENV': '/opt/MVS/lib'
                                }}
                ]
            )
        else:
            return ComposableNode(
                package='mindvision_camera',
                plugin='mindvision_camera::MVCameraNode',
                name='camera_driver',
                parameters=[get_bringup_params('camera_driver')],
                extra_arguments=[{'use_intra_process_comms': True}]
            )

    def create_camera_detector_container(context):
        image_source = LaunchConfiguration('image_source').perform(context)
        image_source = image_source.lower() if image_source else 'video'

        image_node = _make_camera_composable_node(image_source)
        composable_nodes = [image_node, armor_detector_node]

        container = ComposableNodeContainer(
            name='camera_detector_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=composable_nodes,
            output='both',
            emulate_tty=True,
        )
        return [container]

    # 当不使用容器模式时，相机和检测节点均作为独立 Node 直接运行
    def create_standalone_nodes(context):
        image_source = LaunchConfiguration('image_source').perform(context)
        image_source = image_source.lower() if image_source else 'video'

        cam_cfg = get_camera_node_config(image_source)
        det_cfg = get_detector_node_config()

        camera_node = Node(
            package=cam_cfg['package'],
            executable=cam_cfg['executable'],
            name=cam_cfg['name'],
            parameters=cam_cfg.get('parameters', []),
            output='both',
            emulate_tty=True,
        )
        detector_node = Node(
            package=det_cfg['package'],
            executable=det_cfg['executable'],
            name=det_cfg['name'],
            parameters=det_cfg.get('parameters', []),
            output='both',
            emulate_tty=True,
        )
        return [camera_node, detector_node]

    # ==================== 串口节点 ====================
    def create_serial_node_action(context):
        virtual_serial = LaunchConfiguration('virtual_serial').perform(context).lower() == 'true'
        if virtual_serial:
            return [Node(
                package='rm_serial_driver',
                executable='virtual_serial_node',
                name='virtual_serial',
                output='both',
                emulate_tty=True,
                parameters=[get_bringup_params('virtual_serial')],
            )]
        else:
            return [Node(
                package='rm_serial_driver',
                executable='rm_serial_driver_node',
                name='serial_driver',
                output='both',
                emulate_tty=True,
                parameters=[get_bringup_params('serial_driver')],
            )]

    # ==================== 延迟启动配置 ====================
    delay_serial_node = TimerAction(
        period=1.5,
        actions=[OpaqueFunction(function=create_serial_node_action)],
    )
    delay_ballistic_solver = TimerAction(
        period=2.0,
        actions=[ballistic_solver_node],
    )
    # 相机+检测器 — 容器模式或独立节点模式
    if use_container:
        delay_camera_detector = TimerAction(
            period=2.0,
            actions=[OpaqueFunction(function=create_camera_detector_container)],
        )
        delay_standalone = None
    else:
        delay_camera_detector = None
        delay_standalone = TimerAction(
            period=2.0,
            actions=[OpaqueFunction(function=create_standalone_nodes)],
        )

    # 统一 pipeline 节点 — 替代原来 2.5s / 3.0s / 3.5s 三个节点
    delay_gimbal_pipeline = TimerAction(
        period=2.5,
        actions=[gimbal_pipeline_node],
    )

    # ==================== 命名空间 ====================
    push_namespace = PushRosNamespace(LaunchConfiguration('namespace'))

    # ==================== 构建启动描述 ====================
    launch_actions = [
        declare_image_source,
        declare_virtual_serial,
        declare_debug,
        declare_namespace,
        declare_detector_type,
        declare_use_container,
        # declare_enable_auto_buff,

        robot_gimbal_publisher,
        push_namespace,

        delay_serial_node,
        delay_ballistic_solver,
    ]

    if use_container:
        launch_actions.append(delay_camera_detector)
    else:
        launch_actions.append(delay_standalone)

    launch_actions.append(delay_gimbal_pipeline)
    launch_actions.append(buff_detector_node)
    launch_actions.append(buff_pose_estimator_node)

    return LaunchDescription(launch_actions)
