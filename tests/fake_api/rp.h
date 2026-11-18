// Offline state-machine simulator only. Not a hardware implementation.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>
#include <algorithm>
enum { RP_OK, RP_CH_1, RP_WAVEFORM_SINE, RP_GEN_MODE_BURST,
       RP_GEN_TRIG_SRC_INTERNAL, RP_OUT, RP_LOW, RP_HIGH,
       RP_TRIG_SRC_NOW, RP_TRIG_SRC_AWG_PE,
       RP_DIO0_P, RP_DIO1_P, RP_DIO2_P, RP_DIO3_P, RP_DIO4_P };
inline bool output=false, started=false, full=false, armed=false;
inline int shots=0, decimation=1;
inline int16_t samples[16384]{};
inline const char* rp_GetError(int) { return "Simulated API failure"; }
inline int rp_InitReset(bool) { return RP_OK; }
inline int rp_Release() {
    std::fprintf(stderr, "SIM_RELEASE output=%d shots=%d\n", output, shots);
    return RP_OK;
}
inline int rp_GenOutDisable(int) { output=false; return RP_OK; }
inline int rp_GenOutEnable(int) { output=true; return RP_OK; }
inline int rp_GenResetChannelSM(int) { return RP_OK; }
inline int rp_GenWaveform(int,int) { return RP_OK; }
inline int rp_GenFreq(int,float) { return RP_OK; }
inline int rp_GenAmp(int,float) { return RP_OK; }
inline int rp_GenOffset(int,float) { return RP_OK; }
inline int rp_GenMode(int,int) { return RP_OK; }
inline int rp_GenBurstCount(int,int) { return RP_OK; }
inline int rp_GenBurstRepetitions(int,int) { return RP_OK; }
inline int rp_GenBurstPeriod(int,float) { return RP_OK; }
inline int rp_GenSetInitGenValue(int,float) { return RP_OK; }
inline int rp_GenBurstLastValue(int,float) { return RP_OK; }
inline int rp_GenTriggerSource(int,int) { return RP_OK; }
inline int rp_DpinSetDirection(int,int) { return RP_OK; }
inline int rp_DpinSetState(int,int) { return RP_OK; }
inline int rp_AcqReset() { return RP_OK; }
inline int rp_AcqSetDecimationFactor(int value) { decimation=value; return RP_OK; }
inline int rp_AcqSetGain(int,int) { return RP_OK; }
inline int rp_AcqSetAveraging(bool) { return RP_OK; }
inline int rp_AcqSetTriggerDelay(int) { return RP_OK; }
inline int rp_AcqAxiGetMemoryRegion(uint32_t* address,uint32_t* bytes) {
    *address=0; *bytes=32768; return RP_OK;
}
inline int rp_AcqAxiSetDecimationFactor(int) { return RP_OK; }
inline int rp_AcqAxiSetBufferSamples(int,uint32_t,uint32_t) { return RP_OK; }
inline int rp_AcqAxiSetTriggerDelay(int,int) { return RP_OK; }
inline int rp_AcqAxiEnable(int,bool) { return RP_OK; }
inline int rp_AcqGetSamplingRateHz(float* value) { *value=125e6/decimation; return RP_OK; }
inline int rp_AcqStart() { started=true; full=armed=false; return RP_OK; }
inline int rp_AcqStop() { started=false; return RP_OK; }
inline int rp_AcqAxiGetBufferFillState(int,bool* value) { *value=full; return RP_OK; }
inline int rp_AcqGetBufferFillState(bool* value) { *value=full; return RP_OK; }
inline int rp_AcqSetTriggerSrc(int trigger) {
    if (!started) return 1;
    armed=trigger==RP_TRIG_SRC_AWG_PE;
    if (trigger==RP_TRIG_SRC_NOW) { std::fill_n(samples,16384,0); full=true; }
    return RP_OK;
}
inline int rp_GenTriggerOnly(int) {
    if (!output || !started || !armed || std::getenv("RP_FAKE_FAIL_TRIGGER")) return 1;
    ++shots;
    for (int i=0;i<16384;++i) samples[i]=i%1024;
    full=true; armed=false; return RP_OK;
}
inline int rp_AcqAxiGetWritePointer(int,uint32_t* pos) { *pos=16383; return RP_OK; }
inline int rp_AcqAxiGetDataRawDirect(int,uint32_t,uint32_t count,std::vector<std::span<int16_t>>* data) {
    data->push_back(std::span<int16_t>(samples,count)); return RP_OK;
}
inline int rp_AcqAxiGetDataRaw(int,uint32_t,uint32_t* size,int16_t* data) {
    std::copy_n(samples,*size,data); return RP_OK;
}
inline int rp_AcqGetOldestDataRawWithCalib(int,uint32_t* size,int16_t* data) {
    std::copy_n(samples,*size,data); return RP_OK;
}
