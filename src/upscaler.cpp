#include "upscaler.h"
#include "ggml_extend.hpp"
#include "model.h"
#include "stable-diffusion.h"
#include "util.h"

#include <cstdint>
#include <memory>
#include <new>
#include <utility>

namespace {

thread_local sd_upscaler_execution_evidence_t* active_upscaler_evidence = nullptr;

class UpscalerEvidenceScope {
public:
    explicit UpscalerEvidenceScope(sd_upscaler_execution_evidence_t* evidence)
        : previous(active_upscaler_evidence) {
        active_upscaler_evidence = evidence;
    }

    ~UpscalerEvidenceScope() {
        active_upscaler_evidence = previous;
    }

private:
    sd_upscaler_execution_evidence_t* previous;
};

void record_compute_start(bool tile) {
    if (active_upscaler_evidence == nullptr) {
        return;
    }
    ++active_upscaler_evidence->compute_invocation_count;
    if (tile) {
        ++active_upscaler_evidence->tile_compute_invocation_count;
    }
}

void record_compute_result(bool tile, bool succeeded) {
    if (active_upscaler_evidence == nullptr || !succeeded) {
        return;
    }
    ++active_upscaler_evidence->compute_success_count;
    if (tile) {
        ++active_upscaler_evidence->tile_compute_success_count;
    }
}

}  // namespace

UpscalerGGML::UpscalerGGML(int n_threads,
                           bool direct,
                           int tile_size,
                           std::string backend_spec,
                           std::string params_backend_spec)
    : n_threads(n_threads),
      direct(direct),
      tile_size(tile_size),
      backend_spec(std::move(backend_spec)),
      params_backend_spec(std::move(params_backend_spec)) {
}

void UpscalerGGML::set_max_graph_vram_bytes(size_t max_vram_bytes) {
    max_graph_vram_bytes = max_vram_bytes;
    if (esrgan_upscaler) {
        esrgan_upscaler->set_max_graph_vram_bytes(max_vram_bytes);
    }
}

bool UpscalerGGML::load_from_file(const std::string& esrgan_path,
                                  bool offload_params_to_cpu,
                                  int n_threads) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    std::string error;
    if (!backend_manager.init(backend_spec.c_str(),
                              params_backend_spec.c_str(),
                              offload_params_to_cpu,
                              false,
                              false,
                              false,
                              &error)) {
        LOG_ERROR("upscaler backend config failed: %s", error.c_str());
        return false;
    }
    auto backend_for = [&](SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.runtime_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s backend", sd_backend_module_name(module));
        }
        return module_backend;
    };
    auto params_backend_for = [&](SDBackendModule module) {
        ggml_backend_t module_backend = backend_manager.params_backend(module);
        if (module_backend == nullptr) {
            LOG_ERROR("failed to initialize %s params backend", sd_backend_module_name(module));
        }
        return module_backend;
    };
    auto ensure_backend_pair = [&](SDBackendModule module) {
        if (backend_for(module) == nullptr) {
            return false;
        }
        return params_backend_for(module) != nullptr;
    };
    if (!ensure_backend_pair(SDBackendModule::UPSCALER)) {
        return false;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(esrgan_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", esrgan_path.c_str());
        return false;
    }
    model_loader.set_wtype_override(model_data_type);
    LOG_INFO("Upscaler weight type: %s", ggml_type_name(model_data_type));
    esrgan_upscaler = std::make_shared<ESRGAN>(backend_for(SDBackendModule::UPSCALER),
                                               params_backend_for(SDBackendModule::UPSCALER),
                                               tile_size,
                                               model_loader.get_tensor_storage_map());
    esrgan_upscaler->set_max_graph_vram_bytes(max_graph_vram_bytes);
    if (direct) {
        esrgan_upscaler->set_conv2d_direct_enabled(true);
    }
    if (!esrgan_upscaler->load_from_file(esrgan_path, n_threads)) {
        return false;
    }
    return true;
}

sd::Tensor<float> UpscalerGGML::upscale_tensor(const sd::Tensor<float>& input_tensor) {
    sd::Tensor<float> upscaled;
    if (sd_should_cancel()) {
        LOG_WARN("esrgan upscale cancelled before execution");
        return {};
    }
    if (tile_size <= 0 || (input_tensor.shape()[0] <= tile_size && input_tensor.shape()[1] <= tile_size)) {
        record_compute_start(false);
        upscaled = esrgan_upscaler->compute(n_threads, input_tensor);
        record_compute_result(false, !upscaled.empty());
    } else {
        auto on_processing = [&](const sd::Tensor<float>& input_tile) -> sd::Tensor<float> {
            if (sd_should_cancel()) {
                return {};
            }
            record_compute_start(true);
            auto output_tile = esrgan_upscaler->compute(n_threads, input_tile);
            record_compute_result(true, !output_tile.empty());
            if (output_tile.empty() || sd_should_cancel()) {
                LOG_ERROR("esrgan compute failed while processing a tile");
                return {};
            }
            return output_tile;
        };

        upscaled = process_tiles_2d(input_tensor,
                                    static_cast<int>(input_tensor.shape()[0] * esrgan_upscaler->scale),
                                    static_cast<int>(input_tensor.shape()[1] * esrgan_upscaler->scale),
                                    esrgan_upscaler->scale,
                                    tile_size,
                                    tile_size,
                                    0.25f,
                                    false,
                                    false,
                                    on_processing);
    }
    esrgan_upscaler->free_compute_buffer();
    if (upscaled.empty() || sd_should_cancel()) {
        if (sd_should_cancel()) {
            LOG_WARN("esrgan upscale cancelled");
        }
        LOG_ERROR("esrgan compute failed");
        return {};
    }
    return upscaled;
}

sd_image_t UpscalerGGML::upscale(sd_image_t input_image, uint32_t upscale_factor) {
    // The public wrapper validates upscale_factor against the loaded model scale.
    sd_image_t upscaled_image = {0, 0, 0, nullptr};
    int output_width          = (int)input_image.width * esrgan_upscaler->scale;
    int output_height         = (int)input_image.height * esrgan_upscaler->scale;
    LOG_INFO("upscaling from (%i x %i) to (%i x %i)",
             input_image.width, input_image.height, output_width, output_height);

    sd::Tensor<float> input_tensor = sd_image_to_tensor(input_image);
    sd::Tensor<float> upscaled;
    int64_t t0 = ggml_time_ms();
    upscaled   = upscale_tensor(input_tensor);
    if (upscaled.empty()) {
        return upscaled_image;
    }
    sd_image_t upscaled_data = tensor_to_sd_image(upscaled);
    int64_t t3               = ggml_time_ms();
    LOG_INFO("input_image_tensor upscaled, taking %.2fs", (t3 - t0) / 1000.0f);
    upscaled_image = upscaled_data;
    return upscaled_image;
}

struct upscaler_ctx_t {
    UpscalerGGML* upscaler = nullptr;
    sd_upscaler_execution_evidence_t execution_evidence{};
};

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path_c_str,
                                 bool offload_params_to_cpu,
                                 bool direct,
                                 int n_threads,
                                 int tile_size,
                                 const char* backend,
                                 const char* params_backend) {
    std::unique_ptr<upscaler_ctx_t> upscaler_ctx(new (std::nothrow) upscaler_ctx_t());
    if (!upscaler_ctx) {
        return nullptr;
    }
    std::string esrgan_path(esrgan_path_c_str);

    std::unique_ptr<UpscalerGGML> upscaler(new (std::nothrow) UpscalerGGML(
        n_threads, direct, tile_size, SAFE_STR(backend), SAFE_STR(params_backend)));
    if (!upscaler) {
        return nullptr;
    }

    if (!upscaler->load_from_file(esrgan_path, offload_params_to_cpu, n_threads)) {
        return nullptr;
    }
    upscaler_ctx->upscaler = upscaler.release();
    return upscaler_ctx.release();
}

sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor) {
    sd_image_t empty = {0, 0, 0, nullptr};
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr ||
        upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return empty;
    }

    const int native_scale = get_upscale_factor(upscaler_ctx);
    auto& evidence = upscaler_ctx->execution_evidence;
    evidence = {};
    evidence.version = SD_UPSCALER_EXECUTION_EVIDENCE_VERSION;
    evidence.requested_scale = upscale_factor;
    evidence.native_scale = native_scale > 0 ? static_cast<uint32_t>(native_scale) : 0;
    evidence.source_width = input_image.width;
    evidence.source_height = input_image.height;
    evidence.tiled = upscaler_ctx->upscaler->tile_size > 0 &&
                             (input_image.width > static_cast<uint32_t>(upscaler_ctx->upscaler->tile_size) ||
                              input_image.height > static_cast<uint32_t>(upscaler_ctx->upscaler->tile_size))
                         ? 1u
                         : 0u;
    if (native_scale <= 0 || upscale_factor != static_cast<uint32_t>(native_scale)) {
        LOG_ERROR("requested upscale factor %u does not match model scale %d", upscale_factor, native_scale);
        return empty;
    }

    UpscalerEvidenceScope evidence_scope(&evidence);
    sd_image_t result = upscaler_ctx->upscaler->upscale(input_image, upscale_factor);
    evidence.output_width = result.width;
    evidence.output_height = result.height;
    evidence.cancelled = sd_should_cancel() ? 1u : 0u;
    evidence.completed = result.data != nullptr && evidence.cancelled == 0u &&
                                 evidence.compute_invocation_count > 0u &&
                                 evidence.compute_success_count == evidence.compute_invocation_count
                             ? 1u
                             : 0u;
    return result;
}

extern "C" SD_API bool sd_get_upscaler_execution_evidence(
    const upscaler_ctx_t* upscaler_ctx,
    sd_upscaler_execution_evidence_t* evidence) {
    if (upscaler_ctx == nullptr || evidence == nullptr ||
        upscaler_ctx->execution_evidence.version != SD_UPSCALER_EXECUTION_EVIDENCE_VERSION) {
        return false;
    }
    *evidence = upscaler_ctx->execution_evidence;
    return true;
}

int get_upscale_factor(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr || upscaler_ctx->upscaler == nullptr || upscaler_ctx->upscaler->esrgan_upscaler == nullptr) {
        return 1;
    }
    return upscaler_ctx->upscaler->esrgan_upscaler->scale;
}

void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx) {
    if (upscaler_ctx == nullptr) {
        return;
    }
    if (upscaler_ctx->upscaler != nullptr) {
        delete upscaler_ctx->upscaler;
        upscaler_ctx->upscaler = nullptr;
    }
    delete upscaler_ctx;
}
