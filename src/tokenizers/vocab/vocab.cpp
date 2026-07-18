#include "vocab.h"
#include "clip_merges.hpp"
#include "qwen_merges.hpp"
#include "t5.hpp"

#if !defined(__ANDROID__)
#include "gemma2_merges.hpp"
#include "gemma2_vocab.hpp"
#include "gemma_merges.hpp"
#include "gemma_vocab.hpp"
#include "gpt_oss_merges.hpp"
#include "gpt_oss_vocab.hpp"
#include "mistral_merges.hpp"
#include "mistral_vocab.hpp"
#include "umt5.hpp"
#endif

std::string load_clip_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(clip_merges_utf8_c_str), sizeof(clip_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_qwen2_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(qwen2_merges_utf8_c_str), sizeof(qwen2_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string load_mistral_merges() {
#if defined(__ANDROID__)
    return {};
#else
    std::string merges_utf8_str(reinterpret_cast<const char*>(mistral_merges_utf8_c_str), sizeof(mistral_merges_utf8_c_str));
    return merges_utf8_str;
#endif
}

std::string load_mistral_vocab_json() {
#if defined(__ANDROID__)
    return {};
#else
    std::string json_str(reinterpret_cast<const char*>(mistral_vocab_json_utf8_c_str), sizeof(mistral_vocab_json_utf8_c_str));
    return json_str;
#endif
}

std::string load_t5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;
}

std::string load_umt5_tokenizer_json() {
#if defined(__ANDROID__)
    return {};
#else
    std::string json_str(reinterpret_cast<const char*>(umt5_tokenizer_json_str), sizeof(umt5_tokenizer_json_str));
    return json_str;
#endif
}

std::string load_gemma_merges() {
#if defined(__ANDROID__)
    return {};
#else
    std::string merges_utf8_str(reinterpret_cast<const char*>(gemma_merges_utf8_c_str), sizeof(gemma_merges_utf8_c_str));
    return merges_utf8_str;
#endif
}

std::string load_gemma_vocab_json() {
#if defined(__ANDROID__)
    return {};
#else
    std::string json_str(reinterpret_cast<const char*>(gemma_vocab_json_utf8_c_str), sizeof(gemma_vocab_json_utf8_c_str));
    return json_str;
#endif
}

std::string load_gemma2_merges() {
#if defined(__ANDROID__)
    return {};
#else
    std::string merges_utf8_str(reinterpret_cast<const char*>(gemma2_merges_utf8_c_str), sizeof(gemma2_merges_utf8_c_str));
    return merges_utf8_str;
#endif
}

std::string load_gemma2_vocab_json() {
#if defined(__ANDROID__)
    return {};
#else
    std::string json_str(reinterpret_cast<const char*>(gemma2_vocab_json_utf8_c_str), sizeof(gemma2_vocab_json_utf8_c_str));
    return json_str;
#endif
}

std::string load_gpt_oss_merges() {
#if defined(__ANDROID__)
    return {};
#else
    std::string merges_utf8_str(reinterpret_cast<const char*>(gpt_oss_merges_utf8_c_str), sizeof(gpt_oss_merges_utf8_c_str));
    return merges_utf8_str;
#endif
}

std::string load_gpt_oss_vocab_json() {
#if defined(__ANDROID__)
    return {};
#else
    std::string json_str(reinterpret_cast<const char*>(gpt_oss_vocab_json_utf8_c_str), sizeof(gpt_oss_vocab_json_utf8_c_str));
    return json_str;
#endif
}
