# Create GDExtension For ONNX
* I just generated linux version by: "scons -j 18 separate_debug_symbols=yes use_static_cpp=no". ONNXRuntime is here: https://github.com/microsoft/onnxruntime/releases/tag/v1.18.0
* I comment fexception("// throw Ort::Exception(string, code)" is modified around line: 77 and 78) in onnxruntime_cxx_api.h because I am lazy:).
* I removed mac libs because they are more than 50 MB.
 
# Godot RL Agents

This repository contains the Godot 4 asset / plugin for the Godot RL Agents library, you can find out more about the library on its Github page [here](https://github.com/edbeeching/godot_rl_agents).

The Godot RL Agents is a fully Open Source package that allows video game creators, AI researchers and hobbyists the opportunity to learn complex behaviors for their Non Player Characters or agents. 
This libary provided this following functionaly:
* An interface between games created in the [Godot Engine](https://godotengine.org/) and Machine Learning algorithms running in Python
* Wrappers for three well known rl frameworks: StableBaselines3, Sample Factory and [Ray RLLib](https://docs.ray.io/en/latest/rllib-algorithms.html)
* Support for memory-based agents, with LSTM or attention based interfaces
* Support for 2D and 3D games
* A suite of AI sensors to augment your agent's capacity to observe the game world
* Godot and Godot RL Agents are completely free and open source under the very permissive MIT license. No strings attached, no royalties, nothing. 

You can find out more about Godot RL agents in our AAAI-2022 Workshop [paper](https://arxiv.org/abs/2112.03636).

