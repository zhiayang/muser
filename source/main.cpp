// main.cpp
// Copyright (c) 2019, zhiayang
// Licensed under the Apache License Version 2.0.

#include "defs.h"

#include "kissnet.h"

#include <thread>
#include <chrono>
using namespace std::chrono_literals;

#include <sys/mman.h>
#include <arpa/inet.h>
#include <vorbis/codec.h>
#include <vorbis/vorbisenc.h>

// what the fuck? i shouldn't have to do this manually...
extern "C" {
	#include <libavutil/log.h>
	#include <libavutil/timestamp.h>
	#include <libavformat/avformat.h>
	#include <libswresample/swresample.h>
}

constexpr auto OutputSampleRate = 44100;
constexpr auto OutputChannelFmt = AV_CH_LAYOUT_STEREO;
constexpr auto OutputSampleFmt  = AV_SAMPLE_FMT_FLTP;

struct SampleBuffer
{
	uint64_t id = 0;

	float* left  = 0;
	float* right = 0;

	size_t sampleCount = 0;

	// 2mb per channel per buffer. won't be enough for one song, but it's exactly 1 large page.
	// (so, one buffer will be 4mb, since we output stereo)
	static constexpr size_t BufferSize = 2 * 1024 * 1024;
	static constexpr size_t SamplesPerBuffer = BufferSize / sizeof(float);
	static constexpr size_t ChannelCount = 2;

	size_t remaining() const { return SamplesPerBuffer - this->sampleCount; }
	bool isFull() const      { return this->remaining() == 0; }

	void free()
	{
		if(this->left)  munmap(this->left, SampleBuffer::BufferSize);
		if(this->right) munmap(this->right, SampleBuffer::BufferSize);

		this->left = 0;
		this->right = 0;
		this->sampleCount = 0;
	}

	void clear()
	{
		this->left = 0;
		this->right = 0;
		this->sampleCount = 0;
	}

	static SampleBuffer create(size_t sz)
	{
		assert(sz > 0);

		static uint64_t curid = 0;

		float* left = static_cast<float*>(mmap(nullptr, SampleBuffer::BufferSize,
			PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));

		float* right = static_cast<float*>(mmap(nullptr, SampleBuffer::BufferSize,
			PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));

		if(left == reinterpret_cast<float*>(-1) || right == reinterpret_cast<float*>(-1))
			return util::error("out of memory!"), SampleBuffer();

		SampleBuffer ret;
		ret.id = curid++;

		ret.left = left;
		ret.right = right;
		ret.sampleCount = 0;

		return ret;
	}
};

struct RawBuffer
{
	uint64_t id = 0;

	uint8_t* data = 0;
	size_t size = 0;

	size_t capacity = 0;

	static constexpr size_t MinBufferSize = 2 * 1024 * 1024;

	size_t remaining() const { return this->capacity - this->size; }
	bool isFull() const      { return this->remaining() == 0; }

	uint8_t* copyIn(void* ptr, size_t len)
	{
		assert(len <= this->remaining());

		memcpy(this->data + this->size, ptr, len);

		auto ret = this->data + this->size;
		this->size += len;

		return ret;
	}

	uint8_t* pointer()
	{
		return this->data + this->size;
	}

	void free()
	{
		if(this->data)  munmap(this->data, this->capacity);

		this->data = 0;
		this->size = 0;
		this->capacity = 0;
	}

	void clear()
	{
		this->data = 0;
		this->size = 0;
		this->capacity = 0;
	}

	static RawBuffer create(size_t sz)
	{
		sz = std::max(sz, MinBufferSize);

		static uint64_t curid = 0;

		void* data = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

		if(data == reinterpret_cast<void*>(-1))
			return util::error("out of memory!"), RawBuffer();

		RawBuffer ret;

		ret.id = curid++;
		ret.data = (uint8_t*) data;
		ret.size = 0;
		ret.capacity = sz;

		return ret;
	}
};




template <typename T>
struct BufferList
{
	T allocate(size_t sz)
	{
		if(!this->_free.empty() && this->_free.back().remaining() >= sz)
		{
			auto ret = this->_free.back();
			this->_free.pop_back();

			return ret;
		}

		// time to make ):
		auto foo = T::create(sz);
		this->_buffers.push_back(foo);

		return foo;
	}

	void free(T buf)
	{
		std::remove_if(this->_buffers.begin(), this->_buffers.end(), [&buf](auto x) -> bool {
			return x.id == buf.id;
		});

		buf.free();
		buf.clear();

		this->_free.push_back(buf);
	}

	void clearAll()
	{
		for(auto& buf : this->_buffers)
			this->free(buf);
	}

	T* get(size_t atLeast)
	{
		if(this->_buffers.empty() || this->_buffers.back().remaining() < atLeast)
			this->allocate(atLeast);

		return &this->_buffers.back();
	}

	size_t count()
	{
		return this->_buffers.size();
	}

	std::vector<T>& bufs()
	{
		return this->_buffers;
	}

private:
	std::vector<T> _buffers;
	std::vector<T> _free;
};




static BufferList<SampleBuffer> extractSamplesFromFile(const std::fs::path& path)
{
	auto ctx = avformat_alloc_context();
	defer(avformat_free_context(ctx));

	BufferList<SampleBuffer> bufferList;

	// thank the comma operator. you'll see it a lot more. sadly it doesn't wanna work with { },
	// so we gotta return the (empty) sample buffer.
	if(avformat_open_input(&ctx, path.string().c_str(), nullptr, nullptr) < 0)
		return util::error("failed to open input file '%s'", path.string()), bufferList;

	defer(avformat_close_input(&ctx));

	if(avformat_find_stream_info(ctx, nullptr) < 0)
		return util::error("failed to read streams"), bufferList;


	AVStream* strm = 0;
	for(unsigned int i = 0; i < ctx->nb_streams; i++)
	{
		if(ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
		{
			strm = ctx->streams[i];
			break;
		}
	}

	if(!strm) return util::error("no audio stream"), bufferList;

	auto decoder = avcodec_find_decoder(strm->codecpar->codec_id);
	if(!decoder) return util::error("no codec to decode stream"), bufferList;

	auto dctx = avcodec_alloc_context3(decoder);
	if(!dctx) return util::error("failed to allocate decoder context"), bufferList;

	defer(avcodec_free_context(&dctx));

	avcodec_parameters_to_context(dctx, strm->codecpar);
	if(avcodec_open2(dctx, decoder, nullptr) < 0)
		return util::error("failed to open decoder codec"), bufferList;

	// order for this is (output, input)!
	SwrContext* swr = swr_alloc_set_opts(NULL,
		/* output: */ OutputChannelFmt, OutputSampleFmt, OutputSampleRate,
		/* input: */  strm->codecpar->channel_layout, (AVSampleFormat) strm->codecpar->format, strm->codecpar->sample_rate,
		/* log stuff: */ 0, NULL
	);

	if(!swr || swr_init(swr) < 0)
		return util::error("failed to create swresampler!"), bufferList;

	defer(swr_free(&swr));

	{
		AVFrame* frame = av_frame_alloc();
		AVPacket packet;
		av_init_packet(&packet);

		auto copy_samples = [&bufferList](SwrContext* swr, AVFrame* frame) {

			SampleBuffer* sambuf = bufferList.get(/* atLeast: */ 1);

			uint8_t* chans[2] = {
				(uint8_t*) (sambuf->left + sambuf->sampleCount),
				(uint8_t*) (sambuf->right + sambuf->sampleCount)
			};

			auto converted = swr_convert(swr, chans, sambuf->remaining(),
				(const uint8_t**) frame->extended_data, frame->nb_samples);

			if(converted < 0)
			{
				util::error("conversion error");
			}
			else
			{
				sambuf->sampleCount += converted;
				// zpr::println("%zu", sambuf->sampleCount);
			}
		};

		while(true)
		{
			if(av_read_frame(ctx, &packet) < 0)
				break;

			if(packet.stream_index != strm->index)
				continue;

			if(auto s = avcodec_send_packet(dctx, &packet); s < 0)
			{
				if(s == AVERROR(EAGAIN)) continue;
				else                     return util::error("stream error: %d", s), bufferList;
			}

			if(auto r = avcodec_receive_frame(dctx, frame); r < 0)
			{
				if(r == AVERROR(EAGAIN)) continue;
				else                     return util::error("stream error: %d", r), bufferList;
			}

			copy_samples(swr, frame);

			av_frame_unref(frame);
		}

		// flush the buffer.
		avcodec_send_packet(dctx, nullptr);
		while(avcodec_receive_frame(dctx, frame) != AVERROR_EOF)
		{
			copy_samples(swr, frame);
			av_frame_unref(frame);
		}

		av_frame_free(&frame);
	}

	zpr::println("%zu buffers", bufferList.count());
	return bufferList;
}


static void hexdump(uint8_t* arr, size_t len)
{
	constexpr int ValuesPerRow = 8;

	auto iszero = [](uint8_t* ptr, size_t len) -> bool {
		for(size_t i = 0; i < len; i++)
			if(ptr[i]) return false;

		return true;
	};


	int all0sCnt = 0;
	for(size_t i = 0; (len - i >= ValuesPerRow) && (i < len); i += ValuesPerRow)
	{
		if(all0sCnt > 0)
		{
			while((len - ValuesPerRow - i >= ValuesPerRow) && (i < len - ValuesPerRow) && iszero(arr + i, ValuesPerRow))
				i += ValuesPerRow;

			printf("    *\n");
		}

		printf("%5zx:  ", i);
		for(size_t k = 0; k < ValuesPerRow; k++)
			printf(" %02x", arr[i + k]);

		printf("    |");

		for(size_t k = 0; k < ValuesPerRow; k++)
		{
			auto c = arr[i + k];
			(c >= 32 && c <= 127) ? putchar(c) : putchar('.');
		}

		printf("|\n");

		if(iszero(arr + i, ValuesPerRow))
			all0sCnt++;

		else
			all0sCnt = 0;
	}


	if(auto rem = len % ValuesPerRow; rem > 0)
	{
		auto tmp = len - (len % ValuesPerRow);

		printf("%5zx:  ", tmp);
		for(size_t i = 0; i < rem; i++)
			printf(" %02x", arr[tmp + i]);

		for(size_t i = 0; i < (ValuesPerRow - rem); i++)
			printf("   ");

		printf("    |");
		for(size_t i = 0; i < rem; i++)
			(arr[tmp + i] >= 32 && arr[tmp + i] <= 127) ? putchar(arr[tmp + i]) : putchar('.');

		printf("|\n");
	}
}


std::string base64_encode(const uint8_t* src, size_t len)
{
	constexpr unsigned char base64_table[65] =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	size_t olen = 4 * ((len + 2) / 3); /* 3-byte blocks to 4-byte */

	if(olen < len)
		return "";

	std::string outStr;
	outStr.resize(olen);

	uint8_t* out = (uint8_t*) &outStr[0];
	uint8_t* pos = out;

	const uint8_t* end = src + len;
	const uint8_t* in = src;
	while(end - in >= 3)
	{
		*pos++ = base64_table[in[0] >> 2];
		*pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
		*pos++ = base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
		*pos++ = base64_table[in[2] & 0x3f];
		in += 3;
	}

	if(end - in)
	{
		*pos++ = base64_table[in[0] >> 2];
		if(end - in == 1)
		{
			*pos++ = base64_table[(in[0] & 0x03) << 4];
			*pos++ = '=';
		}
		else
		{
			*pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
			*pos++ = base64_table[(in[1] & 0x0f) << 2];
		}
		*pos++ = '=';
	}

	return outStr;
}

static std::string getBase64EncodedConfigString(const ogg_packet& hdr_ident,
	const ogg_packet& hdr_comment, const ogg_packet& hdr_setup)
{
	size_t header_length = hdr_ident.bytes + hdr_comment.bytes + hdr_setup.bytes;

	int sizeSize1 = hdr_ident.bytes < 128 ? 1 : hdr_ident.bytes < 16384 ? 2 : 3;
	int sizeSize2 = hdr_comment.bytes < 128 ? 1 : hdr_comment.bytes < 16384 ? 2 : 3;

	int size1 = hdr_ident.bytes;
	int size2 = hdr_comment.bytes;

	uint32_t config_ident = 0xF05B15;

	size_t total_length = header_length
		+ sizeSize1 + sizeSize2
		+ 4     // number of packed headers field   (32 bits)
		+ 3     // vorbis ident field               (24 bits)
		+ 2     // vorbis length field              (16 bits)
		+ 1;    // number of headers field          (8 bits)


	uint8_t* buf = new uint8_t[total_length];

	// number of packed headers = 1
	*(uint32_t*) &buf[0] = htonl(1);
	buf[4] = (config_ident & 0xFF0000) >> 16;
	buf[5] = (config_ident & 0xFF00) >> 8;
	buf[6] = (config_ident & 0xFF) >> 0;
	*(uint16_t*) &buf[7] = htons(header_length);
	buf[9] = 3 - 1;


	uint8_t* ptr = &buf[10];
	if(size1 >= 16384)  // flag, but no more, because we know length1 <= 32767
		*ptr++ = 0x80;

	if(size1 >= 128)    // flag + the second 7 bits
		*ptr++ = 0x80 | ((size1 & 0x3F80) >> 7);

	 // the low 7 bits
	*ptr++ = size1 & 0x7F;

	// second one:
	if(size2 >= 16384)
		*ptr++ = 0x80;

	if(size2 >= 128)
		*ptr++ = 0x80 | ((size2 & 0x3F80) >> 7);

	*ptr++ = size2 & 0x7F;



	// ok now copy the headers.
	memcpy(ptr, hdr_ident.packet, hdr_ident.bytes);
	ptr += hdr_ident.bytes;

	memcpy(ptr, hdr_comment.packet, hdr_comment.bytes);
	ptr += hdr_comment.bytes;

	memcpy(ptr, hdr_setup.packet, hdr_setup.bytes);
	ptr += hdr_setup.bytes;


	// ok, we're done. base64-encode it:
	auto b64_enc = base64_encode(buf, total_length);
	delete[] buf;

	return b64_enc;
}



static bool processOneFile(const std::fs::path& path)
{
	struct VorbisPacket
	{
		void* data;
		size_t size;
		uint64_t timestamp;
	};

	// auto samples = extractSamplesFromFile(path);

	auto vorbisPkts = std::vector<VorbisPacket>();
	auto buffers = BufferList<RawBuffer>();

	double time_per_granule = 1.0 / (44100.0);

	// load from file.
	#if 1
	{
		auto dump = fopen("vorbis.dump", "rb");
		while(!feof(dump))
		{
			uint64_t sz = 0;
			if(fread(&sz, sizeof(uint64_t), 1, dump) < 0)
				break;

			uint64_t ts = 0;
			fread(&ts, sizeof(uint64_t), 1, dump);

			auto rbuf = buffers.get((size_t) sz);
			assert(rbuf->remaining() >= sz);

			fread(rbuf->pointer(), 1, sz, dump);

			vorbisPkts.push_back(VorbisPacket {
				.data = rbuf->pointer(),
				.size = sz,
				.timestamp = ts
			});

			rbuf->size += sz;
		}

		zpr::println("read %zu buffers", buffers.count());
	}
	#endif


	// vorbis
	#if 0
	{
		vorbis_info vb_info;
		vorbis_info_init(&vb_info);

		// int r = vorbis_encode_init(&vb_info, /* channels: */ 2, /* sample_rate: */ 44100,
		// 	/* max bitrate: */ 256*1000, /* avg: bitrate: */ 192*1000, /* min bitrate: */ 128*1000);

		int r = vorbis_encode_init_vbr(&vb_info, /* channels: */ 2, /* sample_rate: */ 44100, /* quality:*/ 0.4);

		assert(r == 0);

		vorbis_dsp_state vb_dsp;
		vorbis_analysis_init(&vb_dsp, &vb_info);

		vorbis_comment vb_comment;
		vorbis_comment_init(&vb_comment);
		vorbis_comment_add(&vb_comment, "kekw");

		ogg_packet hdr_ident, hdr_comment, hdr_setup;
		vorbis_analysis_headerout(&vb_dsp, &vb_comment, &hdr_ident, &hdr_comment, &hdr_setup);

		size_t totalSamples = 0;

		time_per_granule = vorbis_granule_time(&vb_dsp, 1);

		// get the long block size.
		size_t maxBlockSize = vorbis_info_blocksize(&vb_info, 1);
		util::log("max block size: %zu, time_per_granule: %.8f s", maxBlockSize, time_per_granule);

		zpr::println("config: %s", getBase64EncodedConfigString(hdr_ident, hdr_comment, hdr_setup));

		// just send the whole thing in at once.
		for(auto buf : samples.bufs())
		{
			float** samples = vorbis_analysis_buffer(&vb_dsp, buf.sampleCount);
			assert(samples);

			for(size_t i = 0; i < buf.sampleCount; i++)
			{
				samples[0][i] = buf.left[i];
				samples[1][i] = buf.right[i];

				totalSamples++;
			}

			vorbis_analysis_wrote(&vb_dsp, buf.sampleCount);
		}

		// throw in one more empty one.
		vorbis_analysis_wrote(&vb_dsp, 0);

		fprintf(stderr, "\nfinal samples: %zu\n", totalSamples);

		// just use one block for now
		vorbis_block vb_blk;
		vorbis_block_init(&vb_dsp, &vb_blk);

		size_t watermark1 = 0;
		size_t watermark2 = 0;
		size_t copied = 0;
		while(vorbis_analysis_blockout(&vb_dsp, &vb_blk) == 1)
		{
			// we got data in the block.
			vorbis_analysis(&vb_blk, nullptr);

			vorbis_bitrate_addblock(&vb_blk);

			ogg_packet pkt;
			while(vorbis_bitrate_flushpacket(&vb_dsp, &pkt))
			{
				watermark1 = pkt.granulepos;
				watermark2 = pkt.packetno;

				if(watermark2 % 50 == 0)
					fprintf(stderr, "\x1b[1K\x1b[1Gwm1: %zu, wm2: %zu", watermark1, watermark2);

				// copy it.
				size_t remaining = pkt.bytes;
				assert(remaining <= maxBlockSize);

				void* buf = new uint8_t[maxBlockSize];
				memcpy(buf, pkt.packet, pkt.bytes);

				buffers.push_back(VorbisPacket {
					.data = buf,
					.size = (size_t) pkt.bytes,
					.timestamp = (uint32_t) pkt.granulepos
				});

				// for now:
				assert(pkt.bytes <= 1024);

				copied += pkt.bytes;
			}
		}

		vorbis_block_clear(&vb_blk);


		zpr::println("\nwm1: %d, wm2: %d, copied: %zu", watermark1, watermark2, copied);

		vorbis_comment_clear(&vb_comment);
		vorbis_dsp_clear(&vb_dsp);
		vorbis_info_clear(&vb_info);
	}
	#endif

	// dump to file
	#if 0
	{
		auto dump = fopen("vorbis.dump", "wb");

		for(auto& buf : buffers)
		{
			uint64_t size = buf.size;
			uint64_t ts = buf.timestamp;

			fwrite(&size, sizeof(uint64_t), 1, dump);
			fwrite(&ts, sizeof(uint64_t), 1, dump);
			fwrite(buf.data, 1, buf.size, dump);

			zpr::println("wrote: %zu", size);
		}

		uint64_t last = 0;
		fwrite(&last, sizeof(uint64_t), 1, dump);

		fclose(dump);
	}
	#endif



	// rtp test
	#if 1
	{
		constexpr double buffered_seconds = 5.0;

		kissnet::udp_socket sock(kissnet::endpoint("127.0.0.1", 9999));

		// fucking bitfields in reverse endianness
		// TODO: make this not a bitfield.
		struct rtp_header
		{
			uint8_t csrc_count  : 4;
			uint8_t extension   : 1;
			uint8_t padding     : 1;
			uint8_t version     : 2;

			uint8_t payload_type: 7;
			uint8_t marker      : 1;

			uint16_t sequence;
			uint32_t timestamp;

			uint32_t sync_src;

		} __attribute__((packed));

		struct vorbis_header
		{
			uint8_t ident1;
			uint8_t ident2;
			uint8_t ident3;
			uint8_t other;

		} __attribute__((packed));

		struct vorbis_packet
		{
			uint16_t length;
			uint8_t data[];

		} __attribute__((packed));

		uint32_t timestamp = 0;
		uint16_t sequence = 9;

		struct rtp_packet
		{
			rtp_header hdr;
			vorbis_header vorb;
			vorbis_packet packets[];

		} __attribute__((packed));

		static_assert(sizeof(rtp_packet) < 1400);

		uint8_t* buf = new uint8_t[1420];
		auto packet = (rtp_packet*) buf;

		packet->hdr.version = 2;
		packet->hdr.padding = 0;
		packet->hdr.extension = 0;
		packet->hdr.csrc_count = 0;

		packet->hdr.marker = 0;
		packet->hdr.payload_type = 97;

		packet->hdr.sequence = htons(sequence);
		packet->hdr.timestamp = htonl(timestamp);

		packet->hdr.sync_src = 13456184;

		constexpr uint32_t config_ident = 0xF05B15;
		packet->vorb.ident1 = (config_ident & 0xFF0000) >> 16;
		packet->vorb.ident2 = (config_ident & 0xFF00) >> 8;
		packet->vorb.ident3 = (config_ident & 0xFF) >> 0;


		uint64_t timestamp_ofs = 0;
		uint64_t prev_timestamp = 0;


		size_t idx = 0;
		while(true)
		{
			// loop:
			if(idx == vorbisPkts.size())
			{
				printf("loop");

				idx = 0;
				timestamp_ofs = timestamp;
				prev_timestamp = 0;
			}

			size_t totalsz = 0;
			std::vector<VorbisPacket> tosend;

			size_t remaining = 1400 - sizeof(rtp_packet);
			while(idx < vorbisPkts.size() && tosend.size() < 15)
			{
				auto& buf = vorbisPkts[idx];
				if(remaining < buf.size)
					break;

				tosend.push_back(buf);

				remaining -= buf.size;
				totalsz += buf.size;

				idx++;
			}

			packet->vorb.other = tosend.size() & 0xF;
			auto ptr = (uint8_t*) packet->packets;

			for(auto& buf : tosend)
			{
				*(uint16_t*) ptr = htons(buf.size);
				memcpy(ptr + 2, buf.data, buf.size);

				ptr += sizeof(int16_t) + buf.size;
			}

			timestamp = timestamp_ofs + tosend.front().timestamp;
			packet->hdr.timestamp = htonl(timestamp);
			packet->hdr.sequence = htons(sequence);

			// send.
			sock.send((const std::byte*) packet, ptr - (uint8_t*) packet);

			auto start = std::chrono::high_resolution_clock::now();

			while(tosend.back().timestamp > prev_timestamp)
			{
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> diff = end - start;

				if(diff.count() >= time_per_granule * (tosend.back().timestamp - prev_timestamp) / 1.3)
					break;
			}

			prev_timestamp = tosend.back().timestamp;
			sequence += 1;
		}
	}
	#endif





	// buffers.clearAll();
	// samples.clearAll();

	util::log("ok");
	return true;
}




int main(int argc, char** argv)
{
	if(argc < 2)
	{
		zpr::println("no files");
		return -1;
	}

	vorbis_info vb_info;
	vorbis_info_init(&vb_info);



	processOneFile(argv[1]);
}



/*
	links:
	https://tools.ietf.org/html/rfc3550#section-5.1
	https://tools.ietf.org/html/rfc5215#section-7.1.1
	https://github.com/geeksville/Micro-RTSP/blob/master/src/CRtspSession.cpp
*/




