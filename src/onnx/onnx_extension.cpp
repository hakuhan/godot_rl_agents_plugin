#include "onnx_extension.h"
#include <onnxruntime_cxx_api.h>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/classes/file_access.hpp>

using namespace onnx;

ONNXExtension::ONNXExtension() : env(ORT_LOGGING_LEVEL_WARNING, "godot_onnx"), session(nullptr), batch_size(1) {}

ONNXExtension::~ONNXExtension() {
    free_disposables();
}

void ONNXExtension::_bind_methods()
{
    godot::ClassDB::bind_method(godot::D_METHOD("initialize", "path", "batch_size"), &ONNXExtension::initialize);
    godot::ClassDB::bind_method(godot::D_METHOD("run_inference", "obs", "state_ins"), &ONNXExtension::run_inference);
    godot::ClassDB::bind_method(godot::D_METHOD("load_model", "model_path"), &ONNXExtension::load_model);
    godot::ClassDB::bind_method(godot::D_METHOD("free_disposables"), &ONNXExtension::free_disposables);
}

int ONNXExtension::initialize(godot::String path, int batch_size) {
    model_path = path;
    this->batch_size = batch_size;
    session_options = MakeConfiguredSessionOptions();
    // session = new Ort::Session(env, model_path.utf8().get_data(), session_options);
    load_model(model_path);
    auto output_metadata = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    return output_metadata[1];
}

godot::Dictionary ONNXExtension::run_inference(godot::Array obs, int state_ins) {
    if (!session) {
        // throw std::runtime_error("Model has not been loaded. Call initialize() first.");
        godot::UtilityFunctions::printerr("Model has not been loaded. Call initialize() first.");
        return godot::Dictionary();
    }

    std::vector<int64_t> input_shape = { batch_size, static_cast<int64_t>(obs.size()) / batch_size };
    std::vector<float> input_tensor_values(obs.size());

    for (int i = 0; i < obs.size(); ++i) {
        input_tensor_values[i] = obs[i];
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    std::vector<int64_t> state_shape = { batch_size };
    std::vector<float> state_tensor_values(batch_size, state_ins);

    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        state_tensor_values.data(),
        state_tensor_values.size(),
        state_shape.data(),
        state_shape.size()
    );

    const char* input_node_names[] = { "obs", "state_ins" };
    const char* output_node_names[] = { "output", "state_outs" };

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(state_tensor));

    auto output_tensors = session->Run(
        Ort::RunOptions{ nullptr },
        input_node_names,
        input_tensors.data(),
        input_tensors.size(),
        output_node_names,
        2
    );

    godot::Dictionary output;
    godot::Array output1_array;
    godot::Array output2_array;

    float* float_array1 = output_tensors[0].GetTensorMutableData<float>();
    float* float_array2 = output_tensors[1].GetTensorMutableData<float>();

    for (size_t i = 0; i < output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        output1_array.append(float_array1[i]);
    }

    for (size_t i = 0; i < output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
        output2_array.append(float_array2[i]);
    }

    output["output"] = output1_array;
    output["state_outs"] = output2_array;

    return output;
}

void ONNXExtension::load_model(godot::String model_path) {
    free_disposables(); // 先释放之前的会话

    // 使用文件FileAccess加载数据，因为使用的是Godot的路径.
    // session = new Ort::Session(env, model_path.utf8().get_data(), session_options);
    godot::Ref<godot::FileAccess> file = godot::FileAccess::open(model_path, godot::FileAccess::ModeFlags::READ);
    godot::PackedByteArray model = file->get_buffer((int)file->get_length());

    this->model_path = model_path;
    session = new Ort::Session(env, model.ptr(), (int)file->get_length(), session_options);
}

void ONNXExtension::free_disposables() {
    if (session) {
        delete session;
        session = nullptr;
    }
}
