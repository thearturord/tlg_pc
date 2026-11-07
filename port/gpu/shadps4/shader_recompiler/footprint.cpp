// tlg: shader read footprint, see footprint.h
#include <algorithm>
#include <map>
#include <mutex>
#include <fmt/format.h>
#include "shader_recompiler/footprint.h"
#include "shader_recompiler/info.h"

namespace Shader::Footprint {
namespace {

struct BufferFp {
    u32 sharp_first_dword{};
    bool sharp_in_user_data{};
    u32 const_lo{~0u}, const_end{0};
    u32 const_accesses{}, dynamic_accesses{}, writes{};
    bool formatted{}, const_buffer{};
    u32 widest{};
};

struct AttrFp {
    u32 sgpr_base{}, dword_offset{}, num_elements{}, stride{}, element_bytes{}, step_rate{};
};

struct ShaderFp {
    u64 program{};
    u32 stage{};
    std::map<u32, BufferFp> buffers;
    std::map<u32, AttrFp> attributes;
};

std::mutex g_mutex;
std::map<std::pair<u64, u32>, ShaderFp> g_shaders; // (program hash, hw stage)

ShaderFp& Get(const Info& info) {
    ShaderFp& fp = g_shaders[{info.pgm_hash, static_cast<u32>(info.hw_stage)}];
    fp.program = info.pgm_base >> 8;
    fp.stage = static_cast<u32>(info.hw_stage);
    return fp;
}

} // namespace

void NoteBufferAccess(const Info& info, u32 binding, u32 sharp_first_dword, bool sharp_in_user_data, u32 lo, u32 bytes, bool dynamic, bool write, bool formatted, bool const_buffer) {
    std::scoped_lock lock{g_mutex};
    BufferFp& b = Get(info).buffers[binding];
    b.sharp_first_dword = sharp_first_dword;
    b.sharp_in_user_data = sharp_in_user_data;
    b.formatted |= formatted;
    b.const_buffer |= const_buffer;
    b.widest = std::max(b.widest, bytes);
    if (write) {
        ++b.writes;
    }
    if (dynamic) {
        ++b.dynamic_accesses;
        return;
    }
    ++b.const_accesses;
    b.const_lo = std::min(b.const_lo, lo);
    b.const_end = std::max(b.const_end, lo + bytes);
}

void NoteVertexAttribute(const Info& info, u32 semantic, u32 sgpr_base, u32 dword_offset, u32 num_elements, u32 stride, u32 element_bytes, u32 step_rate) {
    std::scoped_lock lock{g_mutex};
    Get(info).attributes[semantic] = AttrFp{sgpr_base, dword_offset, num_elements, stride, element_bytes, step_rate};
}

nlohmann::json Report(const std::string& hash_filter) {
    std::scoped_lock lock{g_mutex};
    nlohmann::json out = nlohmann::json::object();
    out["shaders_listed"] = g_shaders.size();
    out["note"] = "const_range: bytes inside one buffer record that constant-offset accesses touch; dynamic_accesses: loads whose offset has a register part (range unknown). program: code address >> 8 (the key of /stats shader_names)";
    auto& list = out["shaders"] = nlohmann::json::array();
    for (const auto& [key, fp] : g_shaders) {
        const std::string hash = fmt::format("{:016x}", key.first);
        if (!hash_filter.empty() && hash.find(hash_filter) == std::string::npos) {
            continue;
        }
        nlohmann::json s;
        s["hash"] = hash;
        s["program"] = fmt::format("{:#x}", fp.program);
        s["hw_stage"] = fp.stage;
        auto& buffers = s["buffers"] = nlohmann::json::array();
        for (const auto& [binding, b] : fp.buffers) {
            nlohmann::json j;
            j["binding"] = binding;
            j[b.sharp_in_user_data ? "sharp_user_data_dword" : "sharp_note"] = b.sharp_in_user_data ? nlohmann::json(b.sharp_first_dword) : nlohmann::json("inline or derived");
            if (b.const_accesses != 0) {
                j["const_range"] = fmt::format("{:#x}..{:#x}", b.const_lo, b.const_end);
            }
            j["const_accesses"] = b.const_accesses;
            j["dynamic_accesses"] = b.dynamic_accesses;
            j["widest_access_bytes"] = b.widest;
            if (b.writes != 0) {
                j["writes"] = b.writes;
            }
            if (b.formatted) {
                j["formatted"] = true;
            }
            if (b.const_buffer) {
                j["flat_const_buffer"] = true;
            }
            buffers.push_back(std::move(j));
        }
        auto& attrs = s["vertex_attributes"] = nlohmann::json::array();
        for (const auto& [semantic, a] : fp.attributes) {
            attrs.push_back({{"semantic", semantic}, {"sgpr_base", a.sgpr_base}, {"v_sharp_dword_offset", a.dword_offset}, {"elements", a.num_elements}, {"stride", a.stride},
                             {"element_bytes", a.element_bytes}, {"step_rate", a.step_rate}});
        }
        list.push_back(std::move(s));
    }
    return out;
}

} // namespace Shader::Footprint
