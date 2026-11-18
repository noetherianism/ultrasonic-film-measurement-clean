// Single-client acquisition; optional synchronized CH1 ultrasound bursts.
#include <rp.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cerrno>
#include <thread>
#include <span>
#include <csignal>
#include <algorithm>

static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
static void check(int result) {
    if (result != RP_OK) throw std::runtime_error(rp_GetError(result));
}
static void send_all(int fd, const void* data, size_t count) {
    const char* p = static_cast<const char*>(data);
    while (count) {
        auto n = send(fd, p, count, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("TCP send failed");
        p += n; count -= n;
    }
}
int main(int argc, char** argv) {
    double duration = argc > 1 ? std::atof(argv[1]) : 30;
    bool direct = argc > 2 && std::strcmp(argv[2], "direct") == 0;
    bool dma = direct || (argc > 2 && std::strcmp(argv[2], "dma") == 0);
    double target_fps = argc > 3 ? std::atof(argv[3]) : 0;
    bool ultrasound = argc > 4 && std::strcmp(argv[4], "ultrasound") == 0;
    double frequency = argc > 5 ? std::atof(argv[5]) : 5000000;
    int cycles = argc > 6 ? std::atoi(argv[6]) : 1;
    double amplitude = argc > 7 ? std::atof(argv[7]) : 1.0;
    int decimation = argc > 8 ? std::atoi(argv[8]) : 1;
    bool mux = argc > 9 && std::strcmp(argv[9], "single-ch1") == 0;
    bool live = argc > 10 && std::strcmp(argv[10], "live") == 0;
    bool live_tx = false;
    if (!std::isfinite(target_fps) || target_fps < 0) return 2;
    if (!std::isfinite(duration) || duration < 0 || (!live && duration == 0) || duration > 120) return 2;
    if (live && (!ultrasound || !direct)) return 2;
    if (decimation != 1 && decimation != 2 && decimation != 4 && decimation != 8) return 2;
    if (ultrasound && (!dma || target_fps <= 0 || !std::isfinite(frequency) ||
        frequency < 1 || frequency > 50000000 || cycles < 1 || cycles > 1000 ||
        !std::isfinite(amplitude) || amplitude <= 0 || amplitude > 1.0 ||
        cycles/frequency >= 1.0/target_fps)) return 2;
    if (mux && !ultrasound) return 2;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGHUP, on_signal);
    bool initialized = false;
    int listener = -1, client = -1, rc = 0;
    try {
        check(rp_InitReset(false)); initialized = true;
        if (ultrasound) {
            check(rp_GenOutDisable(RP_CH_1));
            check(rp_GenResetChannelSM(RP_CH_1));
            check(rp_GenWaveform(RP_CH_1, RP_WAVEFORM_SINE));
            check(rp_GenFreq(RP_CH_1, frequency));
            check(rp_GenAmp(RP_CH_1, amplitude));
            check(rp_GenOffset(RP_CH_1, 0));
            check(rp_GenMode(RP_CH_1, RP_GEN_MODE_BURST));
            check(rp_GenBurstCount(RP_CH_1, cycles));
            check(rp_GenBurstRepetitions(RP_CH_1, 1));
            check(rp_GenBurstPeriod(RP_CH_1, std::ceil(cycles/frequency*1e6)+1));
            check(rp_GenSetInitGenValue(RP_CH_1, 0));
            check(rp_GenBurstLastValue(RP_CH_1, 0));
            check(rp_GenTriggerSource(RP_CH_1, RP_GEN_TRIG_SRC_INTERNAL));
            if (mux) {
                for (auto pin : {RP_DIO0_P, RP_DIO1_P, RP_DIO2_P, RP_DIO3_P, RP_DIO4_P}) {
                    check(rp_DpinSetDirection(pin, RP_OUT));
                    check(rp_DpinSetState(pin, pin == RP_DIO4_P ? RP_HIGH : RP_LOW));
                }
            }
        }
        check(rp_AcqReset());
        check(rp_AcqSetDecimationFactor(decimation));
        if (ultrasound) check(rp_AcqSetGain(RP_CH_1, RP_LOW));
        check(rp_AcqSetAveraging(false));
        check(rp_AcqSetTriggerDelay(8192));
        if (dma) {
            uint32_t address=0, bytes=0;
            check(rp_AcqAxiGetMemoryRegion(&address,&bytes));
            if (bytes < 32768 || address % 4096) throw std::runtime_error("Invalid DMA memory region");
            check(rp_AcqAxiSetDecimationFactor(decimation));
            check(rp_AcqAxiSetBufferSamples(RP_CH_1,address,16384));
            check(rp_AcqAxiSetTriggerDelay(RP_CH_1,16384));
            check(rp_AcqAxiEnable(RP_CH_1,true));
        }
        float sample_rate = 0;
        check(rp_AcqGetSamplingRateHz(&sample_rate));
        if (ultrasound && (cycles/frequency >= 16384.0/sample_rate ||
                           16384.0/sample_rate >= 1.0/target_fps))
            throw std::runtime_error("Burst must fit capture window, and capture window must fit frame period");
        // One complete, newly triggered full buffer for every sent frame.
        std::vector<int16_t> data(16384);
        bool validate = true;
        double acquisition = 0, readout = 0, transfer = 0;
        auto acquire = [&]() {
            auto t0 = Clock::now();
            check(rp_AcqStart());
            bool cleared = true;
            check(dma ? rp_AcqAxiGetBufferFillState(RP_CH_1,&cleared)
                      : rp_AcqGetBufferFillState(&cleared));
            if (cleared) throw std::runtime_error("Fill flag did not reset for fresh acquisition");
            if (interrupted) throw std::runtime_error("Interrupted");
            bool transmit = ultrasound && (!live || live_tx);
            check(rp_AcqSetTriggerSrc(transmit ? RP_TRIG_SRC_AWG_PE : RP_TRIG_SRC_NOW));
            if (transmit) check(rp_GenTriggerOnly(RP_CH_1));
            bool full = false;
            do {
                check(dma ? rp_AcqAxiGetBufferFillState(RP_CH_1,&full)
                          : rp_AcqGetBufferFillState(&full));
                if (seconds(t0, Clock::now()) > 5) throw std::runtime_error("Fill timeout");
                if (interrupted) throw std::runtime_error("Interrupted");
            } while (!full);
            auto t1 = Clock::now();
            uint32_t size = data.size();
            if (dma) {
                check(rp_AcqStop());
                uint32_t pos=0;
                check(rp_AcqAxiGetWritePointer(RP_CH_1,&pos));
                pos = (pos+1)%16384;
                if (direct) {
                    std::vector<std::span<int16_t>> regions;
                    check(rp_AcqAxiGetDataRawDirect(RP_CH_1,pos,size,&regions));
                    size_t copied=0;
                    for (auto region: regions) {
                        if (copied+region.size()>data.size()) throw std::runtime_error("DMA span overflow");
                        std::memcpy(data.data()+copied,region.data(),region.size_bytes());
                        copied+=region.size();
                    }
                    if (copied != data.size()) throw std::runtime_error("Incomplete DMA spans");
                    if (validate) {
                        std::vector<int16_t> reference(16384);
                        check(rp_AcqAxiGetDataRaw(RP_CH_1,pos,&size,reference.data()));
                        if (reference != data) {
                            for (size_t j=0;j<data.size();j++) if (reference[j]!=data[j]) {
                                fprintf(stderr,"Mismatch at %zu: direct=%d reference=%d\n",j,data[j],reference[j]); break;
                            }
                            throw std::runtime_error("Direct DMA differs from reference API");
                        }
                    }
                } else {
                    check(rp_AcqAxiGetDataRaw(RP_CH_1,pos,&size,data.data()));
                }
            } else {
                check(rp_AcqGetOldestDataRawWithCalib(RP_CH_1, &size, data.data()));
            }
            if (size != data.size()) throw std::runtime_error("Incomplete frame");
            auto t2 = Clock::now();
            acquisition += seconds(t0,t1); readout += seconds(t1,t2);
        };
        if (ultrasound && !live) check(rp_GenOutEnable(RP_CH_1));
        for (int i=0;i<(live ? 0 : 5);i++) {
            acquire();
            if (ultrasound) std::this_thread::sleep_for(std::chrono::duration<double>(1.0/target_fps));
        }
        validate = false;
        if (direct && !live) fprintf(stderr,"PASS: 5 fresh DMA frames bit-exact against reference API\n");
        acquisition = readout = 0;
        listener = socket(AF_INET,SOCK_STREAM,0);
        if (listener < 0) throw std::runtime_error("socket failed");
        int one = 1;
        setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
        sockaddr_in address{}; address.sin_family=AF_INET; address.sin_port=htons(5001);
        address.sin_addr.s_addr=htonl(INADDR_ANY);
        if (bind(listener,(sockaddr*)&address,sizeof(address)) || listen(listener,1))
            throw std::runtime_error("Cannot bind port 5001");
        fprintf(stderr,"READY port=5001 samples=16384 channels=1 mode=%s int16_LE duration=%.1f sample_rate=%.0f trigger=%s\n",dma ? "DMA_RAW" : "calibrated_RAW",duration,sample_rate,live ? "LIVE" : (ultrasound ? "AWG_PE" : "NOW"));
        pollfd waitfd{listener,POLLIN,0};
        if (poll(&waitfd,1,60000) != 1) throw std::runtime_error("No client within 60 seconds");
        client=accept(listener,nullptr,nullptr);
        if (client < 0) throw std::runtime_error("accept failed");
        setsockopt(client,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
        timeval timeout{5,0};
        setsockopt(client,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        setsockopt(client,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        char go;
        if (recv(client,&go,1,MSG_WAITALL)!=1 || (live ? (go!='0' && go!='1') : go!='G')) throw std::runtime_error("Missing start handshake");
        if (live) {
            live_tx = go == '1';
            if (live_tx) check(rp_GenOutEnable(RP_CH_1));
        }
        uint32_t sequence=0;
        auto start=Clock::now();
        auto next_shot=start;
        auto within_duration = [&]() { return (live && duration == 0) || seconds(start,Clock::now()) < duration; };
        while(within_duration()) {
            if (interrupted) throw std::runtime_error("Interrupted");
            if (live) {
                char commands[64];
                auto count = recv(client, commands, sizeof(commands), MSG_DONTWAIT);
                if (count == 0) throw std::runtime_error("Live receiver disconnected");
                if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                    throw std::runtime_error("Live control receive failed");
                bool requested = live_tx, quit = false;
                for (ssize_t i=0;i<count;i++) {
                    if (commands[i] == 'Q') quit = true;
                    else if (commands[i] == '0' || commands[i] == '1') requested = commands[i] == '1';
                    else throw std::runtime_error("Invalid live control");
                }
                if (quit) break;
                if (requested != live_tx) {
                    check(requested ? rp_GenOutEnable(RP_CH_1) : rp_GenOutDisable(RP_CH_1));
                    live_tx = requested;
                }
                validate = sequence < 5;
            }
            if (target_fps > 0) {
                while (Clock::now() < next_shot && within_duration()) {
                    if (interrupted) throw std::runtime_error("Interrupted");
                    std::this_thread::sleep_until(std::min(next_shot, Clock::now()+std::chrono::milliseconds(1)));
                }
                if (!within_duration()) break;
            }
            acquire();
            auto t=Clock::now();
            uint32_t magic = live ? (live_tx ? 0x52554C31 : 0x52554C30) : (direct ? 0x52505831 : (dma ? 0x52504431 : 0x52504631));
            uint32_t header[]={htonl(magic),htonl(sequence),htonl(16384),htonl(32768)};
            send_all(client,header,sizeof(header));
            send_all(client,data.data(),32768);
            transfer+=seconds(t,Clock::now());
            sequence++;
            if (live && sequence == 5) fprintf(stderr,"PASS: first 5 live DMA frames bit-exact against reference API\n");
            if (target_fps > 0) {
                auto period=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0/target_fps));
                next_shot+=period;
                // Never issue catch-up pulses after a missed frame deadline.
                if (next_shot < Clock::now()) next_shot=Clock::now()+period;
            }
        }
        if (ultrasound) check(rp_GenOutDisable(RP_CH_1));
        uint32_t end[]={htonl(0x52504531),htonl(sequence),0,0};
        send_all(client,end,sizeof(end));
        double elapsed=seconds(start,Clock::now());
        fprintf(stderr,"DONE frames=%u elapsed=%.6f fps=%.3f acquisition_ms=%.6f readout_ms=%.6f send_ms=%.6f\n",
                sequence,elapsed,sequence/elapsed,sequence ? acquisition*1000/sequence : 0,sequence ? readout*1000/sequence : 0,sequence ? transfer*1000/sequence : 0);
    } catch(const std::exception& e) { fprintf(stderr,"ERROR: %s\n",e.what()); rc=1; }
    if (client>=0) close(client);
    if (listener>=0) close(listener);
    if (initialized) { if (ultrasound) rp_GenOutDisable(RP_CH_1); rp_AcqStop(); if (dma) rp_AcqAxiEnable(RP_CH_1,false); rp_Release(); }
    return rc;
}
