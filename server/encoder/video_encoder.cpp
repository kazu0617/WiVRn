/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// Include first because of incompatibility between Eigen and X includes
#include "driver/wivrn_session.h"

#include "video_encoder.h"

#include "encoder_settings.h"
#include "os/os_time.h"
#include "util/u_logging.h"
#include "utils/wivrn_trace.h"
#include "wivrn_config.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>

#if WIVRN_USE_NVENC
#include "video_encoder_nvenc.h"
#endif
#if WIVRN_USE_VAAPI
#include "ffmpeg/video_encoder_va.h"
#endif
#if WIVRN_USE_X264
#include "video_encoder_x264.h"
#endif
#if WIVRN_USE_VULKAN_ENCODE
#include "video_encoder_vulkan_h264.h"
#include "video_encoder_vulkan_h265.h"
#endif
#include "video_encoder_raw.h"

namespace wivrn
{

namespace
{
constexpr uint64_t video_pacing_min_bps = 1'000'000;
constexpr uint64_t video_pacing_max_bps = 10'000'000'000;
constexpr uint64_t video_pacing_default_delay_us = 1'000;
constexpr uint64_t video_pacing_max_delay_us = 5'000;
constexpr int64_t video_pacing_deadline_guard_ns = 5'000'000;
constexpr size_t video_pacing_max_frames = 64;
constexpr uint64_t video_pacing_bits_per_second = 8'000'000'000;

struct video_pacing_config
{
	uint64_t bitrate_bps;
	int64_t max_delay_ns;
};

template <typename T>
std::optional<T> parse_unsigned_decimal(const char * raw, T minimum, T maximum)
{
	if (not raw or not *raw)
		return std::nullopt;

	std::string_view text(raw);
	for (char c: text)
		if (c < '0' or c > '9')
			return std::nullopt;

	T value{};
	auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
	if (error != std::errc{} or end != text.data() + text.size() or value < minimum or value > maximum)
		return std::nullopt;

	return value;
}

const std::optional<video_pacing_config> & get_video_pacing_config()
{
	static const std::optional<video_pacing_config> config = []() -> std::optional<video_pacing_config> {
		const char * raw_bps = std::getenv("WIVRN_VIDEO_PACING_BPS");
		if (not raw_bps)
			return std::nullopt;

		auto bitrate = parse_unsigned_decimal(raw_bps, video_pacing_min_bps, video_pacing_max_bps);
		if (not bitrate)
		{
			U_LOG_W("Invalid WIVRN_VIDEO_PACING_BPS; UDP video pacing disabled (expected 1000000..10000000000)");
			return std::nullopt;
		}

		uint64_t max_delay_us = video_pacing_default_delay_us;
		if (const char * raw_delay = std::getenv("WIVRN_VIDEO_PACING_MAX_DELAY_US"))
		{
			auto parsed = parse_unsigned_decimal(raw_delay, uint64_t(1), video_pacing_max_delay_us);
			if (not parsed)
			{
				U_LOG_W("Invalid WIVRN_VIDEO_PACING_MAX_DELAY_US; UDP video pacing disabled (expected 1..5000)");
				return std::nullopt;
			}
			max_delay_us = *parsed;
		}

		return video_pacing_config{
		        .bitrate_bps = *bitrate,
		        .max_delay_ns = int64_t(max_delay_us * 1'000),
		};
	}();

	return config;
}

std::optional<int64_t> to_server_deadline(const clock_offset & offset, XrTime display_time)
{
	if (not offset or display_time <= 0)
		return std::nullopt;

	if ((offset.b > 0 and display_time < std::numeric_limits<int64_t>::min() + offset.b) or
	    (offset.b < 0 and display_time > std::numeric_limits<int64_t>::max() + offset.b))
		return std::nullopt;

	const int64_t deadline = display_time - offset.b;
	if (deadline <= 0)
		return std::nullopt;
	return deadline;
}

class video_pacing_coordinator
{
	struct frame_state
	{
		int64_t start_ns;
		int64_t deadline_ns;
		uint64_t bytes_sent = 0;
	};

	std::mutex mutex;
	std::map<std::pair<uint64_t, XrTime>, frame_state> frames;

public:
	bool pace_shard(uint64_t frame_idx,
	                XrTime display_time,
	                int64_t deadline_ns,
	                size_t serialized_bytes,
	                const video_pacing_config & config)
	{
		const int64_t now_ns = os_monotonic_get_ns();
		int64_t release_at_ns = now_ns;
		{
			std::unique_lock lock(mutex);
			auto key = std::pair{frame_idx, display_time};
			auto it = frames.find(key);
			if (it == frames.end())
			{
				// Drop only expired frame records. If the bounded table is full,
				// fail open for this frame instead of evicting a live shared budget.
				const int64_t stale_deadline_ns = now_ns > video_pacing_deadline_guard_ns
				                                          ? now_ns - video_pacing_deadline_guard_ns
				                                          : 0;
				for (auto stale = frames.begin(); stale != frames.end();)
				{
					if (stale->second.deadline_ns <= stale_deadline_ns)
						stale = frames.erase(stale);
					else
						++stale;
				}
				if (frames.size() >= video_pacing_max_frames)
					return false;

				it = frames.emplace(key, frame_state{
				                                 .start_ns = now_ns,
				                                 .deadline_ns = deadline_ns,
				                                 .bytes_sent = 0,
				                         })
				             .first;
			}

			auto & frame = it->second;
			frame.deadline_ns = std::min(frame.deadline_ns, deadline_ns);

			// Use cumulative serialized application bytes from every stream that
			// shares this frame key. The cap is measured from the first shard.
			const uint64_t capped_bytes = uint64_t(
			        (uint64_t(config.max_delay_ns) * config.bitrate_bps + video_pacing_bits_per_second - 1) /
			        video_pacing_bits_per_second);
			const int64_t scheduled_delay_ns = frame.bytes_sent >= capped_bytes
			                                           ? config.max_delay_ns
			                                           : int64_t((frame.bytes_sent * video_pacing_bits_per_second) / config.bitrate_bps);
			const int64_t scheduled_at_ns = frame.start_ns + scheduled_delay_ns;
			const int64_t deadline_cutoff_ns = frame.deadline_ns - video_pacing_deadline_guard_ns;
			if (scheduled_at_ns > now_ns and scheduled_at_ns <= deadline_cutoff_ns)
				release_at_ns = scheduled_at_ns;

			if (serialized_bytes > std::numeric_limits<uint64_t>::max() - frame.bytes_sent)
				frame.bytes_sent = std::numeric_limits<uint64_t>::max();
			else
				frame.bytes_sent += serialized_bytes;
		}

		// Never hold the shared-state lock while sleeping. OS scheduler overshoot
		// can exceed the nominal cap or deadline reserve by an unbounded amount.
		if (release_at_ns > now_ns)
		{
			std::this_thread::sleep_for(std::chrono::nanoseconds(release_at_ns - now_ns));
			return true;
		}
		return false;
	}
};

video_pacing_coordinator & pacing_coordinator()
{
	static video_pacing_coordinator coordinator;
	return coordinator;
}
} // namespace

video_encoder::sender::sender() :
        thread([this](std::stop_token t) {
	        while (not t.stop_requested())
	        {
		        data * d = nullptr;
		        {
			        std::unique_lock lock(mutex);
			        if (pending.empty())
				        cv.wait_for(lock, std::chrono::milliseconds(100));
			        else
				        d = &pending.front();
		        }
		        if (d and not d->span.empty())
		        {
			        d->encoder->SendData(d->span, true, d->prefer_control);
			        std::unique_lock lock(mutex);
			        pending.pop_front();
			        cv.notify_all();
		        }
	        }
	        std::unique_lock lock(mutex);
	        pending.clear();
	        cv.notify_all();
        })
{
}

void video_encoder::sender::push(data && d)
{
	std::unique_lock lock(mutex);
	pending.push_back(std::move(d));
	cv.notify_all();
}

void video_encoder::sender::wait_idle(video_encoder * encoder)
{
	std::unique_lock lock(mutex);
	while (std::ranges::any_of(pending, [=](auto & data) { return data.encoder == encoder; }))
		cv.wait_for(lock, std::chrono::milliseconds(100));
}

std::shared_ptr<video_encoder::sender> video_encoder::sender::get()
{
	static std::weak_ptr<video_encoder::sender> instance;
	static std::mutex m;
	std::unique_lock lock(m);
	auto s = instance.lock();
	if (s)
		return s;
	s.reset(new video_encoder::sender());
	instance = s;
	return s;
}

std::unique_ptr<video_encoder> video_encoder::create(
        wivrn::vk_bundle & wivrn_vk,
        const encoder_settings & settings,
        uint8_t stream_idx)
{
	using namespace std::string_literals;
	std::unique_ptr<video_encoder> res;
	if (settings.encoder_name == encoder_vulkan)
	{
#if WIVRN_USE_VULKAN_ENCODE
		switch (settings.codec)
		{
			case video_codec::h264:
				res = video_encoder_vulkan_h264::create(wivrn_vk, settings, stream_idx);
				break;
			case video_codec::h265:
				res = video_encoder_vulkan_h265::create(wivrn_vk, settings, stream_idx);
				break;
			case video_codec::av1:
				throw std::runtime_error("av1 not supported for vulkan video encode");
			case video_codec::raw:
				throw std::runtime_error("raw codec only supported on raw encoder");
		}
#else
		throw std::runtime_error("Vulkan video encode not enabled");
#endif
	}
	if (settings.encoder_name == encoder_x264)
	{
#if WIVRN_USE_X264
		res = std::make_unique<video_encoder_x264>(wivrn_vk, settings, stream_idx);
#else
		throw std::runtime_error("x264 encoder not enabled");
#endif
	}
	if (settings.encoder_name == encoder_nvenc)
	{
#if WIVRN_USE_NVENC
		res = std::make_unique<video_encoder_nvenc>(wivrn_vk, settings, stream_idx);
#else
		throw std::runtime_error("nvenc support not enabled");
#endif
	}
	if (settings.encoder_name == encoder_vaapi)
	{
#if WIVRN_USE_VAAPI
		res = std::make_unique<video_encoder_va>(wivrn_vk, settings, stream_idx);
#else
		throw std::runtime_error("vaapi support not enabled");
#endif
	}

	if (settings.encoder_name == encoder_raw)
	{
		res = std::make_unique<video_encoder_raw>(wivrn_vk, settings, stream_idx);
	}

	if (not res)
		throw std::runtime_error("Failed to create encoder " + settings.encoder_name);

	auto wivrn_dump_video = std::getenv("WIVRN_DUMP_VIDEO");
	if (wivrn_dump_video)
	{
		std::string file(wivrn_dump_video);
		file += "-" + std::to_string(stream_idx);
		switch (settings.codec)
		{
			case h264:
				file += ".h264";
				break;
			case h265:
				file += ".h265";
				break;
			case av1:
				file += ".av1";
				break;
			case raw:
				file += ".yuv";
				break;
		}
		res->video_dump.open(file);
	}
	return res;
}

video_encoder::video_encoder(vk_bundle & vk,
                             uint8_t stream_idx,
                             uint32_t target_queue,
                             const encoder_settings & settings,
                             std::unique_ptr<idr_handler> idr,
                             bool async_send) :
        stream_idx(stream_idx),
        target_queue(target_queue),
        need_transfer(not vk.optimal_transfer(vk.queue.family_index, target_queue)),
        bitrate_multiplier(settings.bitrate_multiplier),
        shared_sender(async_send ? sender::get() : nullptr),
        idr(std::move(idr)),
        extent{
                .width = settings.width,
                .height = settings.height,
        }
{
	assert(this->idr);
}

video_encoder::~video_encoder()
{
	if (shared_sender)
		shared_sender->wait_idle(this);
}

void video_encoder::on_feedback(const from_headset::feedback & feedback)
{
	assert(feedback.stream_index == stream_idx);
	idr->on_feedback(feedback);
}

void video_encoder::reset()
{
	idr->reset();
}

void video_encoder::set_bitrate(uint32_t bitrate_bps)
{
	pending_bitrate = bitrate_bps * bitrate_multiplier;
}

void video_encoder::set_framerate(float framerate)
{
	pending_framerate = framerate;
}

void video_encoder::present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo info, uint64_t frame_index)
{
	wivrn::trace::scope trace_present(wivrn::trace::cpu_track::encoder, stream_idx, frame_index, "present_image");
	// Wait for encoder to be done
	present_slot = (present_slot + 1) % num_slots;
	state[present_slot].wait(busy);
	if (idr->should_skip(frame_index))
	{
		state[present_slot] = skip;
		return;
	}
	state[present_slot] = busy;
	return present_image(y_cbcr, info, present_slot, frame_index);
}

void video_encoder::encode(wivrn_session & cnx,
                           const to_headset::video_stream_data_shard::view_info_t & view_info,
                           uint64_t frame_index)
{
	encode_slot = (encode_slot + 1) % num_slots;

	struct idle_setter
	{
		std::atomic_unsigned_lock_free & state;
		~idle_setter()
		{
			state = idle;
			state.notify_all();
		}
	};
	idle_setter i{state[encode_slot]};

	if (state[encode_slot] == skip)
		return;

	if (shared_sender)
		shared_sender->wait_idle(this);
	this->cnx = &cnx;
	clock = cnx.get_offset();
	pacing_display_time = view_info.display_time;
	pacing_deadline_ns.reset();
	if (shared_sender and get_video_pacing_config())
		pacing_deadline_ns = to_server_deadline(clock, pacing_display_time);

	wivrn::trace::scope trace_encode(wivrn::trace::cpu_track::encoder, stream_idx, frame_index, "encode");
	auto encode_begin = os_monotonic_get_ns();
	timing_info = {
	        .encode_begin = clock.to_headset(encode_begin),
	};

	// Prepare the video shard template
	shard.stream_item_idx = stream_idx;
	shard.frame_idx = frame_index;
	shard.shard_idx = 0;
	shard.view_info = view_info;
	shard.timing_info.reset();

	auto data = encode(encode_slot, frame_index);
	if (data)
	{
		timing_info.encode_end = clock.to_headset(os_monotonic_get_ns());
		assert(shared_sender);
		shared_sender->push(std::move(*data));
	}
}

void video_encoder::SendData(std::span<uint8_t> data, bool end_of_frame, bool control)
{
	std::lock_guard lock(mutex);
	if (shard.shard_idx == 0)
	{
		// One SendData call per NAL; span the whole frame, not each call.
		wivrn::trace::cpu_begin(wivrn::trace::cpu_track::network, stream_idx, shard.frame_idx, "SendData");
		timing_info.send_begin = clock.to_headset(os_monotonic_get_ns());
	}
	if (end_of_frame)
	{
		timing_info.send_end = clock.to_headset(os_monotonic_get_ns());
		if (not timing_info.encode_end)
			timing_info.encode_end = timing_info.send_end;
	}
	if (video_dump)
		video_dump.write((char *)data.data(), data.size());

	ssize_t max_payload_size = (cnx->has_stream() and not control) ? to_headset::video_stream_data_shard::max_payload_size : std::numeric_limits<uint32_t>::max();

	auto begin = data.begin();
	auto end = data.end();
	bool pacing_waited = false;
	while (begin != end)
	{
		const size_t payload_size = std::max(0z, max_payload_size - ssize_t(serialized_size(shard.view_info)));
		auto next = std::min(end, begin + payload_size);
		if (next == end)
		{
			if (end_of_frame)
			{
				if (serialized_size(timing_info) + std::distance(begin, next) <= max_payload_size)
					shard.timing_info = timing_info;
				else
					--next;
			}
		}
		shard.payload = {begin, next};
		if (not control and cnx->has_stream() and shared_sender and pacing_deadline_ns)
		{
			if (const auto & config = get_video_pacing_config())
			{
				const bool waited = pacing_coordinator().pace_shard(shard.frame_idx,
				                                                    pacing_display_time,
				                                                    *pacing_deadline_ns,
				                                                    serialized_size(shard),
				                                                    *config);
				pacing_waited = pacing_waited or waited;
			}
		}
		if (end_of_frame and next == end and pacing_waited)
		{
			// Include opt-in pacing waits in the timing sent with the final shard.
			// This field is fixed width, so refreshing it does not change packet size.
			timing_info.send_end = clock.to_headset(os_monotonic_get_ns());
			shard.timing_info = timing_info;
		}
		try
		{
			if (control)
				cnx->send_control(to_headset::video_stream_data_shard{shard});
			else
				cnx->send_stream(to_headset::video_stream_data_shard{shard});
		}
		catch (...)
		{
			// Ignore network errors
		}
		++shard.shard_idx;
		shard.view_info.reset();
		begin = next;
	}
	if (end_of_frame)
		wivrn::trace::cpu_end(wivrn::trace::cpu_track::network, stream_idx, shard.frame_idx, "SendData");
}

} // namespace wivrn
