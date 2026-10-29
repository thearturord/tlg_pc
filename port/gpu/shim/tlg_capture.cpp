// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: screenshots (F12) and short videos (Shift+F12) of the game frame, see tlg_capture.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tlg_capture.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include <vk_mem_alloc.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#include <miniz.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "tlg_overlay.h"

namespace TlgCapture {

namespace {

using Clock = std::chrono::steady_clock;

constexpr u32 VideoSlots = 4;
constexpr u32 VideoMaxWidth = 1920, VideoMaxHeight = 1080;

enum class Kind : u8 { Screenshot, VideoFrame, VideoEnd };

/// A host visible buffer the frame is copied into
struct Slot {
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void* mapped{};
    u64 size{};
    u64 tick{};
    bool busy{};
    Kind kind{};
    u32 width{}, height{};
    vk::Format format{};
    s64 pts_ms{};
};

/// What the writer thread gets
struct Job {
    Kind kind{};
    u32 width{}, height{};
    vk::Format format{};
    s64 pts_ms{};
    std::vector<u8> pixels;
};

std::atomic<bool> screenshot_wanted{false};
std::atomic<double> video_wanted{0.0};

// present thread
const Vulkan::Instance* vk_instance{};
Slot screenshot_slot{};
std::array<Slot, VideoSlots> video_slots{};
bool video_on{};
bool video_ending{};
Clock::time_point video_start{}, video_until{};
u64 video_frames{}, video_dropped{};
VkImage small_image{};
VmaAllocation small_allocation{};
u32 small_width{}, small_height{};
vk::Format small_format{};

// writer thread
std::mutex jobs_mutex;
std::condition_variable jobs_cv;
std::deque<Job> jobs;
std::once_flag writer_started;

std::filesystem::path OutputPath(const char* extension) {
    auto dir = Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char name[64];
    std::strftime(name, sizeof(name), "TLG_%Y%m%d_%H%M%S", &tm);
    auto path = dir / (std::string(name) + extension);
    for (int n = 2; std::filesystem::exists(path, ec) && n < 100; ++n) {
        path = dir / (std::string(name) + "_" + std::to_string(n) + extension);
    }
    return path;
}

/// The pixels as 8 bit B, G, R, A (the frame is B8G8R8A8, R8G8B8A8 or A2R10G10B10)
void ToBgra(const Job& job, std::vector<u8>& out) {
    const size_t count = size_t{job.width} * job.height;
    out.resize(count * 4);
    const u8* in = job.pixels.data();
    switch (job.format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        for (size_t i = 0; i < count; ++i) {
            out[i * 4 + 0] = in[i * 4 + 2];
            out[i * 4 + 1] = in[i * 4 + 1];
            out[i * 4 + 2] = in[i * 4 + 0];
            out[i * 4 + 3] = 255;
        }
        break;
    case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eA2B10G10R10UnormPack32: {
        const bool rgb_high = job.format == vk::Format::eA2R10G10B10UnormPack32; // R in bits 20..29
        for (size_t i = 0; i < count; ++i) {
            u32 v;
            std::memcpy(&v, in + i * 4, 4);
            const u8 hi = static_cast<u8>((v >> 22) & 0xff), mid = static_cast<u8>((v >> 12) & 0xff), lo = static_cast<u8>((v >> 2) & 0xff);
            out[i * 4 + 0] = rgb_high ? lo : hi;
            out[i * 4 + 1] = mid;
            out[i * 4 + 2] = rgb_high ? hi : lo;
            out[i * 4 + 3] = 255;
        }
        break;
    }
    default: // B8G8R8A8
        std::memcpy(out.data(), in, count * 4);
        for (size_t i = 0; i < count; ++i) {
            out[i * 4 + 3] = 255;
        }
        break;
    }
}

void WriteScreenshot(const Job& job) {
    std::vector<u8> bgra;
    ToBgra(job, bgra);
    const size_t count = size_t{job.width} * job.height;
    std::vector<u8> rgb(count * 3);
    for (size_t i = 0; i < count; ++i) {
        rgb[i * 3 + 0] = bgra[i * 4 + 2];
        rgb[i * 3 + 1] = bgra[i * 4 + 1];
        rgb[i * 3 + 2] = bgra[i * 4 + 0];
    }
    size_t length = 0;
    void* png = tdefl_write_image_to_png_file_in_memory_ex(rgb.data(), static_cast<int>(job.width), static_cast<int>(job.height), 3, &length, 3, false);
    if (png == nullptr) {
        LOG_ERROR(Render, "screenshot: PNG encoding failed");
        return;
    }
    const auto path = OutputPath(".png");
    if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
        std::fwrite(png, 1, length, file);
        std::fclose(file);
        LOG_INFO(Render, "screenshot saved: {} ({}x{})", path.string(), job.width, job.height);
        TlgOverlay::SaveToast(4);
    } else {
        LOG_ERROR(Render, "screenshot: cannot write {}", path.string());
    }
    mz_free(png);
}

/// The video encoder (writer thread)
struct Encoder {
    AVFormatContext* format{};
    AVCodecContext* codec{};
    AVStream* stream{};
    SwsContext* sws{};
    AVFrame* frame{};
    AVPacket* packet{};
    std::filesystem::path path;
    s64 last_pts = -1;
    u64 frames{};

    bool Open(u32 width, u32 height) {
        path = OutputPath(".mp4");
        if (avformat_alloc_output_context2(&format, nullptr, "mp4", path.string().c_str()) < 0 || format == nullptr) {
            return false;
        }
        const AVCodec* encoder = avcodec_find_encoder_by_name("libx264");
        if (encoder == nullptr) {
            encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
        }
        if (encoder == nullptr) {
            encoder = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
        }
        if (encoder == nullptr) {
            return false;
        }
        stream = avformat_new_stream(format, nullptr);
        codec = avcodec_alloc_context3(encoder);
        if (stream == nullptr || codec == nullptr) {
            return false;
        }
        codec->width = static_cast<int>(width);
        codec->height = static_cast<int>(height);
        codec->pix_fmt = AV_PIX_FMT_YUV420P;
        codec->time_base = AVRational{1, 1000}; // pts in milliseconds: the frames keep their real timing
        codec->framerate = AVRational{60, 1};
        codec->gop_size = 60;
        codec->bit_rate = 20'000'000;
        if (format->oformat->flags & AVFMT_GLOBALHEADER) {
            codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        av_opt_set(codec->priv_data, "preset", "veryfast", 0);
        av_opt_set(codec->priv_data, "crf", "18", 0);
        if (avcodec_open2(codec, encoder, nullptr) < 0) {
            return false;
        }
        avcodec_parameters_from_context(stream->codecpar, codec);
        stream->time_base = codec->time_base;
        if (!(format->oformat->flags & AVFMT_NOFILE) && avio_open(&format->pb, path.string().c_str(), AVIO_FLAG_WRITE) < 0) {
            return false;
        }
        if (avformat_write_header(format, nullptr) < 0) {
            return false;
        }
        sws = sws_getContext(static_cast<int>(width), static_cast<int>(height), AV_PIX_FMT_BGRA, static_cast<int>(width), static_cast<int>(height), AV_PIX_FMT_YUV420P,
                             SWS_BILINEAR, nullptr, nullptr, nullptr);
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (sws == nullptr || frame == nullptr || packet == nullptr) {
            return false;
        }
        frame->format = codec->pix_fmt;
        frame->width = codec->width;
        frame->height = codec->height;
        return av_frame_get_buffer(frame, 0) >= 0;
    }

    void Drain(AVFrame* input) {
        if (avcodec_send_frame(codec, input) < 0) {
            return;
        }
        while (avcodec_receive_packet(codec, packet) == 0) {
            av_packet_rescale_ts(packet, codec->time_base, stream->time_base);
            packet->stream_index = stream->index;
            av_interleaved_write_frame(format, packet);
        }
    }

    void Add(const Job& job) {
        std::vector<u8> bgra;
        ToBgra(job, bgra);
        if (av_frame_make_writable(frame) < 0) {
            return;
        }
        const u8* src[1] = {bgra.data()};
        const int stride[1] = {static_cast<int>(job.width * 4)};
        sws_scale(sws, src, stride, 0, static_cast<int>(job.height), frame->data, frame->linesize);
        frame->pts = std::max(job.pts_ms, last_pts + 1);
        last_pts = frame->pts;
        Drain(frame);
        ++frames;
    }

    void Close(bool good) {
        if (codec != nullptr && good) {
            Drain(nullptr);
        }
        if (format != nullptr && good) {
            av_write_trailer(format);
        }
        if (format != nullptr && format->pb != nullptr && !(format->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format->pb);
        }
        sws_freeContext(sws);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec);
        avformat_free_context(format);
        sws = nullptr;
        format = nullptr;
        stream = nullptr;
    }
};

void WriterLoop() {
    Encoder encoder;
    bool open = false;
    bool failed = false;
    for (;;) {
        Job job;
        {
            std::unique_lock lock{jobs_mutex};
            jobs_cv.wait(lock, [] { return !jobs.empty(); });
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        switch (job.kind) {
        case Kind::Screenshot:
            WriteScreenshot(job);
            break;
        case Kind::VideoFrame:
            if (!open && !failed) {
                open = encoder.Open(job.width, job.height);
                if (!open) {
                    LOG_ERROR(Render, "video: the encoder could not be opened ({})", encoder.path.string());
                    encoder.Close(false);
                    failed = true;
                }
            }
            if (open) {
                encoder.Add(job);
            }
            break;
        case Kind::VideoEnd:
            if (open) {
                encoder.Close(true);
                LOG_INFO(Render, "video saved: {} ({} frames, {}x{})", encoder.path.string(), encoder.frames, job.width, job.height);
                TlgOverlay::SaveToast(6);
            }
            encoder = Encoder{};
            open = false;
            failed = false;
            break;
        }
    }
}

void Submit(Job&& job) {
    std::call_once(writer_started, [] { std::thread{WriterLoop}.detach(); });
    {
        std::scoped_lock lock{jobs_mutex};
        jobs.push_back(std::move(job));
    }
    jobs_cv.notify_one();
}

bool EnsureSlot(Slot& slot, u64 size) {
    if (slot.buffer != VK_NULL_HANDLE && slot.size >= size) {
        return true;
    }
    if (slot.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(vk_instance->GetAllocator(), slot.buffer, slot.allocation);
        slot = {};
    }
    const VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_info{.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(vk_instance->GetAllocator(), &buffer_info, &alloc_info, &slot.buffer, &slot.allocation, &info) != VK_SUCCESS) {
        slot = {};
        return false;
    }
    slot.mapped = info.pMappedData;
    slot.size = size;
    return slot.mapped != nullptr;
}

bool EnsureSmallImage(u32 width, u32 height, vk::Format format) {
    if (small_image != VK_NULL_HANDLE && small_width == width && small_height == height && small_format == format) {
        return true;
    }
    if (small_image != VK_NULL_HANDLE) {
        // only while no video copy is in flight (a new size or format starts with all slots free, see Record)
        vmaDestroyImage(vk_instance->GetAllocator(), small_image, small_allocation);
        small_image = VK_NULL_HANDLE;
    }
    const VkImageCreateInfo image_info{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                       .imageType = VK_IMAGE_TYPE_2D,
                                       .format = static_cast<VkFormat>(format),
                                       .extent = {width, height, 1},
                                       .mipLevels = 1,
                                       .arrayLayers = 1,
                                       .samples = VK_SAMPLE_COUNT_1_BIT,
                                       .tiling = VK_IMAGE_TILING_OPTIMAL,
                                       .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    const VmaAllocationCreateInfo alloc_info{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_instance->GetAllocator(), &image_info, &alloc_info, &small_image, &small_allocation, nullptr) != VK_SUCCESS) {
        small_image = VK_NULL_HANDLE;
        return false;
    }
    small_width = width;
    small_height = height;
    small_format = format;
    return true;
}

void CopyToSlot(vk::CommandBuffer cmdbuf, vk::Image image, u32 width, u32 height, Slot& slot) {
    const vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };
    cmdbuf.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, vk::Buffer{slot.buffer}, region);
}

} // namespace

void RequestScreenshot() {
    screenshot_wanted.store(true, std::memory_order_release);
}

void RequestVideo(double seconds) {
    video_wanted.store(seconds, std::memory_order_release);
}

void Record(const Vulkan::Instance& instance, vk::CommandBuffer cmdbuf, vk::Image image, u32 width, u32 height, vk::Format format, u64 tick) {
    vk_instance = &instance;
    bool copied = false;
    if (screenshot_wanted.exchange(false, std::memory_order_acq_rel)) {
        if (!screenshot_slot.busy && EnsureSlot(screenshot_slot, u64{width} * height * 4)) {
            CopyToSlot(cmdbuf, image, width, height, screenshot_slot);
            screenshot_slot.busy = true;
            screenshot_slot.tick = tick;
            screenshot_slot.kind = Kind::Screenshot;
            screenshot_slot.width = width;
            screenshot_slot.height = height;
            screenshot_slot.format = format;
            copied = true;
        }
    }
    const double wanted = video_wanted.exchange(0.0, std::memory_order_acq_rel);
    const auto now = Clock::now();
    if (wanted > 0.0 && !video_on && !video_ending) {
        video_on = true;
        video_start = now;
        video_until = now + std::chrono::milliseconds(static_cast<s64>(wanted * 1000.0));
        video_frames = 0;
        video_dropped = 0;
        TlgOverlay::SaveToast(5);
        LOG_INFO(Render, "video: recording {:.1f} s", wanted);
    }
    if (video_on && now >= video_until) {
        video_on = false;
        video_ending = true; // the end goes to the writer once every frame copy has been read back (Poll)
    }
    if (video_on) {
        // fit 1920x1080, even sizes (the encoder's 4:2:0)
        const double scale = std::min({1.0, double(VideoMaxWidth) / width, double(VideoMaxHeight) / height});
        const u32 w = std::max<u32>(2, static_cast<u32>(width * scale) & ~1u), h = std::max<u32>(2, static_cast<u32>(height * scale) & ~1u);
        Slot* slot = nullptr;
        for (auto& s : video_slots) {
            if (!s.busy) {
                slot = &s;
                break;
            }
        }
        const bool all_free = std::ranges::none_of(video_slots, [](const Slot& s) { return s.busy; });
        const bool same_image = small_image != VK_NULL_HANDLE && small_width == w && small_height == h && small_format == format;
        if (slot == nullptr || (!same_image && !all_free) || !EnsureSmallImage(w, h, format) || !EnsureSlot(*slot, u64{w} * h * 4)) {
            ++video_dropped;
        } else {
            const vk::ImageSubresourceRange range{.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
            const vk::ImageMemoryBarrier to_dst{.srcAccessMask = vk::AccessFlagBits::eTransferRead,
                                                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                                                .oldLayout = vk::ImageLayout::eUndefined,
                                                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                .image = vk::Image{small_image},
                                                .subresourceRange = range};
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, to_dst);
            const vk::ImageBlit blit{
                .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(width), static_cast<s32>(height), 1}},
                .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{static_cast<s32>(w), static_cast<s32>(h), 1}},
            };
            cmdbuf.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, vk::Image{small_image}, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
            const vk::ImageMemoryBarrier to_src{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                                                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                                                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                .image = vk::Image{small_image},
                                                .subresourceRange = range};
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, to_src);
            CopyToSlot(cmdbuf, vk::Image{small_image}, w, h, *slot);
            slot->busy = true;
            slot->tick = tick;
            slot->kind = Kind::VideoFrame;
            slot->width = w;
            slot->height = h;
            slot->format = format;
            slot->pts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - video_start).count();
            ++video_frames;
            copied = true;
        }
    }
    if (copied) {
        // the copies are read by the host once the submission is done
        const vk::MemoryBarrier to_host{.srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eHostRead};
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, to_host, {}, {});
    }
}

void Poll(Vulkan::Scheduler& scheduler) {
    // self test (TLG_CAPTURE_TEST=N): a screenshot and a 5 second video N seconds after the first frame
    static const double test_after = [] {
        const char* e = std::getenv("TLG_CAPTURE_TEST");
        return e != nullptr && *e != 0 ? std::atof(e) : 0.0;
    }();
    static const Clock::time_point first_poll = Clock::now();
    static bool test_done = false;
    if (test_after > 0.0 && !test_done && Clock::now() - first_poll >= std::chrono::milliseconds(static_cast<s64>(test_after * 1000.0))) {
        test_done = true;
        RequestScreenshot();
        RequestVideo(5.0);
    }
    if (vk_instance == nullptr) {
        return;
    }
    const auto hand_over = [&](Slot& slot) {
        if (!slot.busy || !scheduler.IsFree(slot.tick)) {
            return;
        }
        vmaInvalidateAllocation(vk_instance->GetAllocator(), slot.allocation, 0, VK_WHOLE_SIZE);
        Job job{.kind = slot.kind, .width = slot.width, .height = slot.height, .format = slot.format, .pts_ms = slot.pts_ms};
        const u8* data = static_cast<const u8*>(slot.mapped);
        job.pixels.assign(data, data + u64{slot.width} * slot.height * 4);
        slot.busy = false;
        Submit(std::move(job));
    };
    hand_over(screenshot_slot);
    // video frames in the order they were taken (slots finish in submission order)
    std::array<Slot*, VideoSlots> order{};
    for (u32 i = 0; i < VideoSlots; ++i) {
        order[i] = &video_slots[i];
    }
    std::ranges::sort(order, [](const Slot* a, const Slot* b) { return a->tick < b->tick; });
    for (Slot* slot : order) {
        if (slot->busy && !scheduler.IsFree(slot->tick)) {
            break; // keep the order: a later one waits for this one
        }
        hand_over(*slot);
    }
    if (video_ending && std::ranges::none_of(video_slots, [](const Slot& s) { return s.busy; })) {
        video_ending = false;
        Submit(Job{.kind = Kind::VideoEnd, .width = small_width, .height = small_height});
        LOG_INFO(Render, "video: {} frames taken, {} dropped (the readback ring was full)", video_frames, video_dropped);
    }
}

} // namespace TlgCapture
