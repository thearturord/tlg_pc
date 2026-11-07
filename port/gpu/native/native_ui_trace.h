// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"

/// UI draw trace (2026-11-07): the title menu entry "Continue" is sometimes not visible (random, any depth). To find the cause, not the effect, every draw of the user interface text
/// (pixel programs named in TLG_UI_TRACE_PS, default bb4d66bf) keeps what it reads from guest memory: each read's kind, address, size, a hash of the bytes and the first 64 bytes as
/// hex and floats (the vertex data of a glyph quad names its screen position, so each draw can be tied to a menu line). The newest 8 presented frames are kept. A frame where the entry
/// is missing can then be compared with one where it shows: which draw, which address, which bytes differ.
/// Switches: TLG_UI_TRACE=1 arms it from the start; `/uitrace?arm=1&ps=hex,hex` and `/uitrace?arm=0` at run time; `/uitrace` returns the JSON; every frame walk freeze writes uitrace.json.
namespace Core::UiTrace {

/// Raster thread: true while a traced draw runs (one atomic read when idle).
[[nodiscard]] bool Active();
/// Anyone: armed at all (cheap check before BeginDraw).
[[nodiscard]] bool Armed();

void Arm(bool on, const std::string& ps_list = {});

/// Raster thread, before the draw's resources are bound. Starts a traced draw when `ps_hash` is in the list.
void BeginDraw(u64 seq, u64 ps_hash, u64 vs_hash, u64 color0, u32 count, u32 instances, u32 caller, u32 frame);
/// Raster thread: a read of guest memory by the traced draw.
void NoteRead(const char* kind, u64 address, u64 size);
/// Raster thread: an image the traced draw samples (guest address, byte size, description: format, size, tile mode, whether it is a render target of other draws).
void NoteImage(u64 address, u64 size, const std::string& description);
/// Raster thread, texture cache refresh: hash of the image's guest memory (0 when not armed or bigger than 256 KB). Call before the bytes are read and after the upload, then NoteImageUpload.
[[nodiscard]] u64 UploadHash(u64 address, u64 size);
void NoteImageUpload(u64 address, u64 size, u64 hash_before, u64 hash_after, bool source_gpu, const std::string& gpu_page_map);
/// Anyone: the buffer cache drops the "GPU wrote these bytes" mark of a range (the next upload of an image there reads guest memory instead of the GPU's data).
void NoteSupersede(u64 address, u64 size, u64 return_address);
/// Raster thread: the draw is done.
void EndDraw();
/// Raster thread: the GPU writes a guest memory range (texture cache invalidation by an image copy / clear / memory copy / written storage buffer or image). Kept only when it overlaps an image a traced draw
/// has sampled: these are the writers of the title menu text pictures (they are all zero in guest memory, the GPU makes them).
void NoteGpuWrite(const char* by, u64 address, u64 size);
/// Raster thread: a memory copy of at least 64 KB on the GPU. The writes of its source (compute output) made before are kept too, so the order writer, copy, picture can be read.
void NoteGpuCopy(u64 dst, u64 src, u64 size);

/// The kept frames as JSON.
[[nodiscard]] std::string Json();

} // namespace Core::UiTrace
