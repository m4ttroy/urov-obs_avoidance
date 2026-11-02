#!/usr/bin/env bash

# Directory location
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_src_dir="$(cd "$script_dir/../../../.." && pwd)"
ARDUPILOT_GAZEBO="${ARDUPILOT_GAZEBO:-$workspace_src_dir/ardupilot_gazebo}"
ARDUPILOT_HOME="${ARDUPILOT_GAZEBO:-$workspace_src_dir/ardupilot}"

export GZ_VERSION=harmonic

# Add results of ArduSub build
export PATH="$ARDUPILOT_HOME/build/sitl/bin:$PATH"

# Navigate to sim_vehicle.py
export PATH="$PATH:${ARDUPILOT_HOME}/ardupilot/Tools/autotest"

# Add bluerov2_gz models and worlds
export GZ_SIM_RESOURCE_PATH="$script_dir/models:$script_dir/worlds:$GZ_SIM_RESOURCE_PATH"

# Add ardupilot_gazebo plugin
export GZ_SIM_SYSTEM_PLUGIN_PATH="$ARDUPILOT_GAZEBO/build:$GZ_SIM_SYSTEM_PLUGIN_PATH"