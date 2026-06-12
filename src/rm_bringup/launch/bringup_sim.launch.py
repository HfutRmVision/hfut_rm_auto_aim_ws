#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation bringup for the gimbal pipeline.

This launch starts the auto-aim algorithm nodes for simulation. Camera drivers,
video players, and real serial drivers are intentionally not started; their data
must be provided by the simulator.
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _load_default_launch_params():
    try:
        config_path = os.path.join(
            get_package_share_directory("rm_bringup"),
            "config",
            "launch_params_decoupled.yaml",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def generate_launch_description():
    launch_params = _load_default_launch_params()
    sim_params = launch_params.get("sim", {})
    if not isinstance(sim_params, dict):
        sim_params = {}

    default_robot = str(
        sim_params.get("robot", launch_params.get("robot", "sim"))
    ).strip() or "sim"
    default_detector_type = str(
        sim_params.get(
            "detector_type",
            launch_params.get("detector_type", "armor_detector"),
        )
    ).strip() or "armor_detector"
    default_debug = str(
        sim_params.get("debug", launch_params.get("debug", True))
    ).lower()
    default_use_sim_time = str(sim_params.get("use_sim_time", True)).lower()
    default_use_detector = str(sim_params.get("use_detector", True)).lower()
    default_start_ballistic_solver = str(
        sim_params.get("start_ballistic_solver", True)
    ).lower()

    declare_args = [
        DeclareLaunchArgument(
            "robot",
            default_value=default_robot,
            description="Robot config name under rm_bringup/config/<robot>.",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value=str(launch_params.get("namespace", "")),
            description="Namespace applied to launched algorithm nodes.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value=default_use_sim_time,
            description="Use simulator /clock.",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value=default_debug,
            description="Enable debug publishers in detector and gimbal pipeline.",
        ),
        DeclareLaunchArgument(
            "use_detector",
            default_value=default_use_detector,
            description=(
                "true: simulator publishes image_raw/camera_info and this launch "
                "starts armor_detector; false: simulator publishes Armors directly."
            ),
        ),
        DeclareLaunchArgument(
            "detector_type",
            default_value=default_detector_type,
            description="Detector type when use_detector=true: armor_detector | armor_detector_nn.",
        ),
        DeclareLaunchArgument(
            "start_ballistic_solver",
            default_value=default_start_ballistic_solver,
            description="Start local ballistic_solver service.",
        ),
        DeclareLaunchArgument(
            "armors_topic",
            default_value=str(
                sim_params.get("armors_topic", "/armor_detector/armors")
            ),
            description=(
                "Armors topic consumed by gimbal_pipeline. Published by detector "
                "when use_detector=true, otherwise by simulator."
            ),
        ),
        DeclareLaunchArgument(
            "cmd_gimbal_topic",
            default_value=str(
                sim_params.get("cmd_gimbal_topic", "/armor_solver/cmd_gimbal")
            ),
            description="Gimbal command topic published to simulator.",
        ),
    ]

    def launch_setup(context, *args, **kwargs):
        del args, kwargs

        robot_name = LaunchConfiguration("robot").perform(context).strip() or default_robot
        namespace = LaunchConfiguration("namespace").perform(context).strip()
        use_detector = _as_bool(LaunchConfiguration("use_detector").perform(context))
        detector_type = LaunchConfiguration("detector_type").perform(context).strip()
        start_ballistic_solver = _as_bool(
            LaunchConfiguration("start_ballistic_solver").perform(context)
        )

        rm_bringup_share = get_package_share_directory("rm_bringup")
        bringup_config_root = os.path.join(rm_bringup_share, "config")
        robot_config_root = os.path.join(bringup_config_root, robot_name)

        def bringup_config_file(filename):
            candidate = os.path.join(robot_config_root, filename)
            if os.path.isfile(candidate):
                return candidate
            fallback = os.path.join(bringup_config_root, filename)
            if os.path.isfile(fallback):
                return fallback
            return None

        def package_config_file(package_name, filename):
            return os.path.join(
                get_package_share_directory(package_name),
                "config",
                filename,
            )

        def node_config_file(filename, package_name, package_filename=None):
            bringup_file = bringup_config_file(filename)
            if bringup_file:
                return bringup_file
            return package_config_file(
                package_name,
                package_filename if package_filename is not None else filename,
            )

        def package_config_with_bringup_override(
            filename, package_name, package_filename=None
        ):
            pkg_file = package_config_file(
                package_name,
                package_filename if package_filename is not None else filename,
            )
            bringup_file = bringup_config_file(filename)
            if bringup_file:
                return [pkg_file, bringup_file]
            return [pkg_file]

        use_sim_time = ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        )

        actions = []

        if use_detector:
            if detector_type == "armor_detector_nn":
                detector_package = "armor_detector_nn"
                detector_executable = "armor_detector_nn_node"
                detector_params = package_config_with_bringup_override(
                    "armor_detector_nn.yaml",
                    "armor_detector_nn",
                    "armor_detector_nn.yaml",
                )
            elif detector_type == "armor_detector":
                detector_package = "armor_detector"
                detector_executable = "armor_detector_node"
                detector_params = [
                    node_config_file(
                        "armor_detector_params.yaml",
                        "armor_detector",
                        "armor_detector.yaml",
                    )
                ]
            else:
                raise RuntimeError(
                    f"Unsupported detector_type '{detector_type}'. "
                    "Use 'armor_detector' or 'armor_detector_nn'."
                )

            detector_params.append({
                "debug": ParameterValue(LaunchConfiguration("debug"), value_type=bool),
                "use_sim_time": use_sim_time,
            })

            actions.append(
                Node(
                    package=detector_package,
                    executable=detector_executable,
                    name="armor_detector",
                    namespace=namespace,
                    output="both",
                    emulate_tty=True,
                    parameters=detector_params,
                    remappings=[
                        ("image_raw", "/image_raw"),
                        ("camera_info", "/camera_info"),
                        ("armor_detector/armors", LaunchConfiguration("armors_topic")),
                    ],
                )
            )

        if start_ballistic_solver:
            actions.append(
                Node(
                    package="ballistic_solver",
                    executable="ballistic_solver_node_exe",
                    name="ballistic_solver",
                    namespace=namespace,
                    output="screen",
                    emulate_tty=True,
                    parameters=[
                        *package_config_with_bringup_override(
                            "ballistic_solver.yaml",
                            "ballistic_solver",
                            "ballistic_solver.yaml",
                        ),
                        {"use_sim_time": use_sim_time},
                    ],
                )
            )

        gimbal_pipeline_params = [
            *package_config_with_bringup_override(
                "gimbal_pipeline.yaml",
                "gimbal_pipeline",
                "gimbal_pipeline.yaml",
            ),
        ]
        gimbal_pipeline_params.append({
            "debug_mode": ParameterValue(LaunchConfiguration("debug"), value_type=bool),
            "use_sim_time": use_sim_time,
        })

        actions.append(
            Node(
                package="gimbal_pipeline",
                executable="gimbal_pipeline_node",
                name="gimbal_pipeline",
                namespace=namespace,
                output="both",
                emulate_tty=True,
                parameters=gimbal_pipeline_params,
                remappings=[
                    ("/armor_detector/armors", LaunchConfiguration("armors_topic")),
                    ("camera_info", "/camera_info"),
                    ("cmd_gimbal", LaunchConfiguration("cmd_gimbal_topic")),
                ],
            )
        )

        return actions

    return LaunchDescription([
        *declare_args,
        OpaqueFunction(function=launch_setup),
    ])
