// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include <nlohmann/json_fwd.hpp>

#include <array>
#include <string>
#include "common/types.h"
#include "video_core/amdgpu/draw_state.h"
#include "core/native_binding.h"

// Host functions called from the guest at chosen game functions (hook mode "call", see guest_hooks.h).
//
// The game's render layer (JsDeviceContext) is a D3D11-style object model; every draw goes through
// PrepareDraw. These handlers read the context's object pointers at the draw entry points. The first
// handlers only measure (how much of the draw state changes from one draw to the next); the native path
// builds on the same reading code. Layout of the context and the objects: documents/tlg/spike/js_ctx_state.h.
namespace Vulkan {
struct FixedState;
}

namespace Core::NativeDraw {

/// Register groups of the command processor's draw state (Liverpool::RegisterGroupOf): the registers of a record are stored by group.
constexpr u32 NumRegisterGroups = 8;

/// Address of the host function registered under `name` (a PS4_SYSV_ABI function taking
/// (a0..a5, return address) and returning non-zero when it handled the call and the guest function must
/// be skipped), or 0.
u64 FindHandler(const std::string& name);

/// Runtime address of the eboot image (callers are classified by their Ghidra address).
/// Draw count of a record whose draw packet is not known yet (matches any count).
constexpr u32 UnknownDrawCount = 0xfffffffe;

/// Cumulative counters of the native path (read by the debug server's /native endpoint; the log lines use their own).
struct LiveCounters {
    std::atomic<u64> gfx_tags{}, gfx_exec{};
    std::atomic<u64> compute_tags{}, compute_exec{};
    std::atomic<u64> asc_exec{}, asc_packet{}, asc_no_record{}, fill_exec{}, mode_switch_exec{}, asc_barrier_exec{}, clear_exec{}, asc_begin_stripped{}, counter_readback{}, async_copy_exec{}, device_begin_stripped{};
    /// synchronization packets taken out of the stream: scratch barrier units (label, end of shader event, wait on the label,
    /// cache acquire), stand-alone cache acquires and no-op events
    /// draws of a shader without a fetch object yet (`native_sdk_strip` bit 2147483648): the game's draw function ran, the record was built again afterwards
    /// the texture map helper's upload dispatch (`map_native`): calls seen, records patched into the tag, refused, executed
    /// compute program audit (`cs_audit`): record dispatches seen, those whose program (queue state plus record) differs from the record alone on a zeroed program, and the dwords that differ
    /// `native_packet_path` 0: command buffers that are not pure and the packets of them that were not executed
    std::atomic<u64> packet_path_skipped_buffers{}, packet_path_skipped_packets{};
    /// register values the native records apply to the command processor's draw state (OverlaySet): offered, not covered by any member, equal to the held value, changed
    std::array<std::atomic<u32>, 65536> overlay_changed_by_index{};
    std::atomic<u64> overlay_group_checked{}, overlay_group_violations{};
    /// cp_group_sig: groups skipped because their identity was already applied, groups applied, skipped groups checked against the record, and violations (by group)
    std::atomic<u64> group_sig_skipped{}, group_sig_applied{}, group_sig_audited{}, group_sig_violations{};
    std::array<std::atomic<u64>, NumRegisterGroups> group_sig_violations_by_group{};
    /// cp_stage_key: draws whose pipeline came from the record's key, draws that went the full way and stored a key, draws without a key, full lookups of
    /// keyed draws (audits) and the ones that differed
    std::atomic<u64> stage_key_hits{}, stage_key_misses{}, stage_key_uncached{}, stage_key_audits{}, stage_key_violations{};
    /// cp_exec_inputs: draws planned from the record alone, draws that could not be (no cached inputs or pipeline yet), draws planned both ways and compared, and the ones that differed
    std::atomic<u64> exec_typed{}, exec_fallback{}, exec_audits{}, exec_violations{};
    /// why a draw was not planned from its record: 0 the switches are not all on, 1 the record has no key (not typed, no fixed state), 2 the command processor is not split,
    /// 3 a fixed-state verify draw, 4 no cached render target inputs or pipeline for its keys yet, 5 a fast clear eliminate that could not be planned
    std::array<std::atomic<u64>, 8> exec_reason{};
    /// of the typed ones: fast clear eliminates and inline draws (the count came with the draw packet)
    std::atomic<u64> exec_fast_clear{}, exec_inline{}, exec_inline_seen{};
    /// the native path's private register block: draws and clears planned from it, plans compared with the shared register image (legacy switch on), differences
    std::atomic<u64> state_planned{}, state_audits{}, state_violations{}, state_group_violations{};
    std::atomic<u64> overlay_offered{}, overlay_unmapped{}, overlay_same{}, overlay_changed{}, overlay_records{};
    std::atomic<u64> cs_audit_dispatches{}, cs_audit_differing{};
    std::array<std::atomic<u64>, 128> cs_audit_dword{};
    std::atomic<u64> map_tried{}, map_patched{}, map_refused{}, map_exec{};
    std::atomic<u64> fetch_rebuild_tried{}, fetch_rebuilt{}, fetch_rebuild_refused{}, fetch_rebuild_exec{};
    std::atomic<u64> sync_units_stripped{}, sync_acquires_stripped{}, sync_events_stripped{}, barrier_calls_skipped{};
    /// direct submission, step 1 (shadow index, `native_sdk_strip` bit 1048576): tags registered at their address, command
    /// buffer spans walked, spans without any packet besides entries ("pure") and with some ("mixed"), tags the walk found that
    /// the index has / has not, indexed tags the walk did not find
    std::atomic<u64> shadow_anchors{}, shadow_spans{}, shadow_pure{}, shadow_mixed{}, shadow_matched{}, shadow_extra{}, shadow_missing{}, shadow_nested{}, shadow_stale{};
    /// sync records (`native_sdk_strip` bit 2097152): packets turned into tags, by kind (1 graphics end-of-pipe, 2 release memory, 3 wait, 4 write data)
    /// and by place (the hook of FUN_02480d60, the sweep when the next tag is written, the submit of a command buffer); packets the
    /// hook left as written; tags the command processor executed, by kind
    std::array<std::atomic<u64>, 5> sync_converted{}, sync_exec{};
    std::atomic<u64> sync_converted_hook{}, sync_converted_sweep{}, sync_converted_submit{}, fence_packets_kept{};
    /// direct submission, step 3a (bit 4194304): graphics buffers whose entry list was built, pure / mixed ones, entries listed, entries the walk
    /// found in the list, entries that differ or are missing, buffers whose non-entry packets differ between list and walk, buffers whose list is incomplete (not compared)
    std::atomic<u64> direct_built{}, direct_pure{}, direct_mixed{}, direct_entries{}, direct_matched{}, direct_entry_mismatch{}, direct_purity_mismatch{}, direct_incomplete{}, direct_executed{}, direct_executed_asc{};
    std::atomic<u64> track_native{}, track_fce{}, track_game_second{}, track_game_depth{};
    /// fixed-function state of native draws: used from the engine's objects, compared with the registers, and the compares that differed
    std::atomic<u64> fixed_native{}, fixed_checked{}, fixed_diff{};
    /// command packets by the group of the last native tag: 0 none, 1 executed draw, 2 packet-path draw, 3 executed
    /// dispatch, 4 packet-path dispatch, 5 fast clear eliminate
    std::array<std::atomic<u64>, 6> packets{};
    std::atomic<u64> asc_packets{};
};
LiveCounters& Live();

/// The engine's JsDevice (0 until known).
u64 EngineDevice();

/// The number the next record will get (records are numbered in the order the game's hooks create them, across all contexts).
u64 NextSeq();

void SetImageBase(VAddr base);
/// Load address of the game's main module (0 until known).
VAddr ImageBase();
/// C4 (update audit): announces the scene's per object constant slots the engine wrote since the last call (see `SceneSlotWrites`); called at the start of a frame.
void FlushSceneSlotAnnouncements();
/// True on a thread that is running a guarded record build: a fault there is handled by the builder, not fatal.
bool IsGuarded();
/// N6.3i: runs `fn(argument)` with the same fault guard as the record builder; false when it faulted.
bool GuardedCall(void (*fn)(void*), void* argument);
/// N6.3i (cascade generator): issues `DrawIndexedCommon(ctx, count, first, 0, 1)` as the depth pass loop does at its call site, through the draw hook (the tag, then the engine's function).
void IssueIndexedDraw(u64 ctx, u64 count, u64 first, u64 site_ghidra, u64 a4, u64 a3 = 0);
/// Same for `DrawIndexedIndirect(ctx, argument buffer object, byte offset)` at the loop's indirect call site.
void IssueIndirectDraw(u64 ctx, u64 buffer, u64 offset, u64 site_ghidra);
/// Census of the draws the generators issue (`TLG_GEN_CENSUS=1`): kind 0 indexed, 1 indirect; calls, ticks in the tag, ticks in the engine draw function, draws the tag took over.
void TagCensusRead(std::array<u64, 5>& ticks, u64& calls);
void UserDataCensusRead(u64& scan_ticks, u64& fill_ticks, u64& tables, u64& slots);
/// N7 groundwork: the shader input table audit of the generated draws (native_draw.cpp, AuditBindings).
void InputTableAuditRead(nlohmann::json& out);
void MergeCensusRead(std::array<u64, 6>& values);
void RecordCacheCensusRead(u64& lookups, u64& hits_1m, u64& hits_64k);
void MakeRecordCensusRead(std::array<u64, 6>& ticks, u64& calls);
void IssueCensusRead(u32 kind, u64& calls, u64& tag_ticks, u64& original_ticks, u64& taken_by_tag);

/// N6.3e (shadow generator): builds the record of a draw from `ctx`, which may be a private copy of the device context patched from the snapshot instead of the live one. Returns
/// the record number, 0 when the build faulted. The record is not tagged into any command stream and nothing refers to it.
u64 ShadowBuildRecord(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3);
/// Compares two records (the one built from the live context and the one built from the shadow context) and appends one line per difference: registers (except the pointers to the
/// per draw tables), bindings, geometry ids, the descriptor key, flags and extents. Returns the number of differences.
u32 CompareRecords(u64 seq_a, u64 seq_b, std::vector<std::string>& differences);

/// Raster thread: compares the register state recorded for tagged draw `seq` with the raster thread's register file
/// (`reg_array`, index = register dword offset) and counts the differences (shadow mode).
void VerifyRecord(u64 seq, const u32* reg_array);
/// The same for the raster thread's draw state: only the registers that belong to the state are compared.
void VerifyRecord(u64 seq, const AmdGpu::DrawRegs& state);

/// Index base of the compute program registers in a record: dword `d` of the compute program is index `ComputeRegisterBase + d`.
constexpr u32 ComputeRegisterBase = 0xE000;

/// Raster thread: the same comparison for a dispatch; `cs_dwords` is the compute program the dispatch runs with.
void VerifyComputeRecord(u64 seq, const u32* cs_dwords);

struct ExecRegister {
    u32 index; ///< index in the register file (dword offset)
    u32 value;
};

/// A draw the game's render layer did not write as command packets: the registers its objects describe and the
/// draw arguments.
struct ExecDraw {
    static constexpr u32 MaxRegisters = 320;
    u64 seq{}; ///< the record this draw was taken from (diagnostics: the native registry audit)
    /// NativeDraw v2: the descriptors of the record in the user data of the two stages (graphics records); `bindings_complete` when none was dropped
    std::array<RecordBinding, MaxRecordBindings> bindings;
    u32 binding_count{};
    bool bindings_complete{};
    bool bindings_overflow{}; ///< the list did not fit (MaxRecordBindings)
    /// N6.2: the engine ids of the buffers the draw reads as vertex or index data (the engine's vertex buffer slots of the input layout, the index buffer view)
    std::array<u64, MaxGeometryIds> geometry_ids{};
    u32 geometry_id_count{};
    std::array<ExecRegister, MaxRegisters> regs;
    u32 count{};
    /// The first `grouped` registers are stored by group (group g is regs[group_begin[g]] to regs[group_begin[g + 1]]), the rest always applies.
    /// `group_sig[g]` identifies what the group's values come from (the engine's objects); 0: no identity, always apply.
    u32 grouped{};
    std::array<u16, NumRegisterGroups + 1> group_begin{};
    std::array<u64, NumRegisterGroups> group_sig{};
    /// Identity of everything the shader variants and the pipeline of the draw depend on besides the fixed-function state: the group identities and the
    /// descriptor fields the stage specializations compare. 0: unknown, the pipeline is looked up from the registers.
    u64 stage_key{};
    /// Hash of the values of the register groups the render target inputs of the plan come from (targets, other state, vertex program), made when the record
    /// was built. 0: unknown.
    u64 inputs_key{};
    /// Typed per draw inputs (TakeExecDraw with `typed`): the user data registers of the vertex and pixel stage, a bit per dword of the stage that the record writes.
    /// They are not among `regs`.
    bool typed{};
    u16 vs_user_data_mask{}, ps_user_data_mask{};
    std::array<u32, 16> vs_user_data{}, ps_user_data{};
    u32 kind{};        ///< 0 = indexed, 1 = auto
    u32 start_index{}; ///< indexed draws
    u64 args_addr{};   ///< indirect draws (kind 2)
    u64 extra_addr{};  ///< copy records (kind 10): the source address
    u32 offset{};
    u32 vertex_sgpr{};
    u32 instance_sgpr{};
    std::array<u32, 8> cb_extent{}; ///< render target extent hints (the game's trailing NOP of a target bind)
    u32 cb_extent_mask{};           ///< bit per slot whose hint is set by this draw
    u32 db_extent{};
    bool has_extents{};
    u32 a1{}; ///< kind 6 (buffer fill): number of dwords; start_index holds the value, args_addr the destination
    bool cb_extents_only{}; ///< fast clear eliminate: only the render target extents change
    u64 fill_destination{}; ///< dispatches of the device's fill shader: the pattern fill it performs (0: not one)
    u32 fill_bytes{};
    u32 fill_dwords{}; ///< dwords of the pattern (at most 4)
    std::array<u32, 4> fill_pattern{};
    u64 copy_destination{}; ///< dispatches of the device's buffer copy shader: the memory copy it performs (null: not one)
    u64 copy_source{};
    u32 copy_bytes{};
    const Vulkan::FixedState* fixed{}; ///< draws: the fixed-function state built from the engine's objects (null: from the registers)
    u32 fixed_stale{};                 ///< FixedStateStale groups of `fixed` the hardware keeps from earlier draws
    u32 dst_usage{~0u};                ///< copy and counter read-back records: Usage of the destination resource (3 staging)
};

/// Command processor: the index / vertex count the record's draw was called with (0xffffffff when the record is gone).
u32 RecordDrawCount(u64 seq);

/// Direct submission, step 1 (documents/tlg/native-direct-submission.md). The record builders register the address of every tag
/// they write; the packet walk of a command buffer span compares what it finds with the registered tags of the span and counts
/// the packets that are not entries (inert padding aside), which tells whether the span could be submitted without the walk.
bool SubmitShadowEnabled();
void NoteAnchor(u64 address, u64 seq, u32 kind);
/// One line that describes the tag at `header` and the record it names (kind, mode, calling function as a Ghidra address, operands). Used by the entry dump.
std::string DescribeEntry(const void* header);
/// Submit time: the synchronization packets of the command buffer (and of the indirect buffers it calls) become tags.
void PrepareSubmittedBuffer(const u32* dcb, u32 dwords);
/// Command processor: when the no-op `tag` stands for a synchronization packet, writes that packet into `packet` and returns its
/// opcode (0x47, 0x49, 0x3c, 0x37); 0 for any other no-op.
u32 RestoreSyncTag(const u32* tag, std::array<u32, 8>& packet, bool count_execution = true);
/// Per opcode: packets in walked spans that are not entries, flips and indirect buffers included (/native).
u64 ShadowNonEntryCount(u32 opcode);

class SubmitShadow {
public:
    SubmitShadow() = default;
    SubmitShadow(const SubmitShadow&) = delete;
    SubmitShadow& operator=(const SubmitShadow&) = delete;
    ~SubmitShadow() {
        End();
    }
    /// `queue`: 0 graphics, 1 + n async compute queue n; the span is the dwords of the command buffer.
    void Begin(u32 queue, const u32* begin, u32 dwords);
    /// The walk found a tag at `header`.
    void Tag(const void* header);
    /// The walk found another packet: padding / no-ops the walk ignores are `inert`.
    void Packet(u32 opcode, bool inert);
    /// The walk follows an indirect buffer: its tags belong to that span, even when it lies inside this one.
    void Indirect(u64 address, u32 dwords);
    /// The walk skipped the payload of a no-op: an old tag left in it (the buffers are reused) is not an entry of this span.
    void Skipped(u64 address, u32 dwords);
    void End();
    bool Active() const {
        return active;
    }

private:
    bool active{};
    u32 queue{};
    u64 begin{};
    u64 end{};
    std::vector<u64> expected; ///< sorted tag addresses of the span
    std::vector<u8> seen;
    std::array<u32, 256> other{}; ///< packets by opcode that are not entries
    u32 extra{};
    std::vector<std::pair<u64, u64>> nested;  ///< address ranges of the indirect buffers the walk followed
    std::vector<std::pair<u64, u64>> skipped; ///< payloads of the no-ops the walk stepped over
};
/// Tags the walk found without an index entry, by tag kind (/native).
u64 ShadowExtraByKind(u32 kind);

/// The Ghidra address of the call that made record `seq` (0 when the record is gone).
u64 RecordCaller(u64 seq);
/// Diagnostics: the index of the first register of the record with this value (0xffffffff none) and the number of registers the record carries in `count`.
u32 RecordRegisterWith(u64 seq, u32 value, u32* count);
/// Diagnostics: the vertex / pixel user data registers the record writes (values in `vs` / `ps`), returned as masks vs | ps << 16 (0xffffffff: record gone).
u32 RecordUserData(u64 seq, u32* vs, u32* ps);

/// Command processor: the record of a draw the hook executed natively. False when the record was overwritten.
bool TakeExecDraw(u64 seq, ExecDraw& out, bool typed = false);

/// The constants of a synthesized compute record live in the table arena, in host memory. The GPU reads buffer data through guest addresses
/// below 2^40 and the guest memory backing, so the descriptor of such a constant buffer holds an address in a window of unused guest
/// addresses; CopyFromArenaWindow answers reads of it.
bool CopyFromArenaWindow(u64 address, u8* dest, u64 size);
/// The range lies in that window (it is not a mapping of the runtime: no clamping to a mapping).
bool IsArenaWindow(u64 address, u64 size);

/// The record is a copy into a staging resource: the image it writes is returned to guest memory (explicit readback of an image).
bool RecordReadsBack(u64 seq);

/// The first register of the user data of the vertex / pixel stage in the register file (16 dwords each).
constexpr u32 VsUserDataRegister = 0x2C00 + 0x4c;
constexpr u32 PsUserDataRegister = 0x2C00 + 0x0c;

/// The register effect of a constant command packet sequence of the emulated Gnm driver (the hardware default state the game
/// sends at the start of every command buffer): the writes in order, parsed once. A tag record names the template; the
/// command processor applies the writes in place of the packets.
struct StateWrite {
    enum Kind : u8 { Context, Sh, Uconfig, NumInstances, ClearState, IndexType };
    u8 kind;
    u32 reg;   ///< register offset within its range
    u32 value;
};
struct StateTemplate {
    std::vector<StateWrite> writes;
};

/// The template of `prefix` followed by `sequence` (both packet dwords); the same pointers always give the same template.
/// Null when the packets contain anything but register writes, clear state and no-ops.
const StateTemplate* HardwareStateTemplate(const u32* prefix, u32 prefix_dwords, const u32* sequence, u32 sequence_dwords);

/// The template of a small indirect buffer that holds nothing but register writes, clear state and padding (the game records its
/// default state once and submits it every frame). Cached by the content; null for anything else.
const StateTemplate* IndirectStateTemplate(const u32* packets, u32 dwords);

} // namespace Core::NativeDraw
