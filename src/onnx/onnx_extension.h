#ifndef ONNX_EXTENSION_H
#define ONNX_EXTENSION_H

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <onnxruntime_cxx_api.h>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <vector>

namespace onnx
{
    class ONNXExtension : public godot::Object 
    {
        GDCLASS(ONNXExtension, godot::Object)

        enum ComputeName {
            CUDA,
            ROCm,
            DirectML,
            CoreML,
            CPU
        };

    protected:
        static void _bind_methods();

    public:
        ONNXExtension();
        ~ONNXExtension();

        int initialize(godot::String path, int batch_size);
        void load_model(godot::String model_path);
        godot::Dictionary run_inference(godot::Array obs, int state_ins);
        void free_disposables();

    public:
        static Ort::SessionOptions MakeConfiguredSessionOptions() {
            Ort::SessionOptions session_options;
            ApplySystemSpecificOptions(session_options);
            return session_options;
        }

        static void ApplySystemSpecificOptions(Ort::SessionOptions& session_options) {
            godot::String OSName = godot::OS::get_singleton()->get_name();
            ComputeName ComputeAPI = ComputeCheck();

            godot::UtilityFunctions::print("OS: ", OSName, " Compute API: ", ComputeAPI);

            // Temporarily using CPU on all platforms to avoid errors detected with DML
            // ComputeName ComputeAPI = CPU;
        }

        static ComputeName ComputeCheck(){
            godot::String adapterName = godot::RenderingServer::get_singleton()->get_video_adapter_name();
            adapterName = adapterName.to_upper();

            if (adapterName.contains("INTEL")) {
                return DirectML;
            } else if (adapterName.contains("AMD") || adapterName.contains("RADEON")) {
                return DirectML;
            } else if (adapterName.contains("NVIDIA")) {
                return CUDA;
            }

            godot::UtilityFunctions::print("Graphics Card not recognized."); // Should use CPU
            return CPU;
        }

    private:
        Ort::Env env;
        Ort::Session* session = nullptr;
        Ort::SessionOptions session_options;
        godot::String model_path;
        int batch_size;
    };

};

#endif // ONNX_EXTENSION_H
