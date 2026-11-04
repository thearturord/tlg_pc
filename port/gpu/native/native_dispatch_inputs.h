// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <bit>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

/// G1 (documents/tlg/native plan/executor-consumer-plan.md): what the rasterizer and the pipeline cache need to know about one compute dispatch, as plain numbers.
/// It replaces `AmdGpu::ComputeProgram` (the PS4's compute register block) in the consumers: the producer (`Liverpool::MakeDispatchInputs`, from the dwords of the dispatch record) is
/// the only place that knows the register layout. The member names are the ones the consumers already used.
namespace Core::NativeDispatch {

struct ThreadCount {
    u16 full{};
    u16 partial{};
};

struct DispatchInputs {
    static constexpr u32 NumUserData = 16;
    /// The dwords of the dispatch record in the register layout (dispatch initiator at 0, user data at 64), kept for the record verify (`VerifyComputeRecord`) only; F6b removes them.
    static constexpr u32 RecordWords = 80;

    u32 dispatch_initiator{};
    u32 dim_x{};
    u32 dim_y{};
    u32 dim_z{};
    ThreadCount num_thread_x{};
    ThreadCount num_thread_y{};
    ThreadCount num_thread_z{};
    u64 address{}; ///< The program address shifted right by 8, as the dispatch record holds it (`Address` shifts it back)
    struct Settings {
        u32 num_vgprs{};
        u32 num_sgprs{};
        u32 fp_round_mode32{};  ///< the numbers of the PS4's modes; the recompiler's own enums are made from them at the pipeline cache's runtime info (G2)
        u32 fp_round_mode64{};
        u32 fp_denorm_mode32{};
        u32 fp_denorm_mode64{};
        u32 dx10_clamp{};
        u32 ieee_mode{};
        u32 scratch_en{};
        u32 num_user_regs{};
        u32 tgid_enable{};
        u32 lds_dwords{};
    } settings{};
    std::array<u32, NumUserData> user_data{};
    std::array<u32, RecordWords> words{};

    template <typename T = u8*>
    const T Address() const {
        return std::bit_cast<T>(address << 8);
    }

    u32 SharedMemSize() const noexcept {
        // lds_dwords is in units of 128 dwords. We return bytes.
        return settings.lds_dwords * 128 * 4;
    }

    u32 NumWorkgroups() const noexcept {
        return dim_x * dim_y * dim_z;
    }

    bool IsTgidEnabled(u32 i) const noexcept {
        return (settings.tgid_enable >> i) & 1;
    }
};

/// Shadow of the producer: every conversion is read back through the accessors and compared with the register block it came from (`g1_dispatch_inputs` in /native).
void NoteChecked(bool equal) noexcept;

[[nodiscard]] nlohmann::json Report();

} // namespace Core::NativeDispatch
