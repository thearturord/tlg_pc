// tlg: shader read footprint (2026-11-07). What bytes of each buffer a shader can touch, found while the shader is recompiled:
// - buffer loads / stores / atomics: the offset inside the buffer record when it is a constant (inst_offset + immediate soffset), or "dynamic" when a register is added;
// - vertex attributes of the fetch shader: the V# that feeds each attribute, its stride and the bytes of one element.
// A shader that comes from the shader cache is not recompiled and is not listed (the cache key hashes the recompiler sources, so the first run after a build recompiles everything).
// Published by the debug server as `/shader_footprint`.
#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "common/types.h"

namespace Shader {
struct Info;
}

namespace Shader::Footprint {

/// One buffer access found in the IR. `lo` is the offset inside the buffer record, `bytes` the size of the access, `dynamic` when a register is part of the offset (the range is then unknown).
void NoteBufferAccess(const Info& info, u32 binding, u32 sharp_first_dword, bool sharp_in_user_data, u32 lo, u32 bytes, bool dynamic, bool write, bool formatted, bool const_buffer);

/// One vertex attribute of a fetch shader.
void NoteVertexAttribute(const Info& info, u32 semantic, u32 sgpr_base, u32 dword_offset, u32 num_elements, u32 stride, u32 element_bytes, u32 step_rate);

/// Everything noted so far (`hash`: only the shaders whose program hash contains this hex text, empty: all)
nlohmann::json Report(const std::string& hash_filter);

} // namespace Shader::Footprint
