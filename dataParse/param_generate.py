"""
    Author      : Shelta Zhao(赵小棠)
    Affiliation : Nanjing University
    Email       : xiaotang_zhao@outlook.com
    Description : This script converts radar data to Point Cloud Data (PCD) format.
"""

import os
import yaml
import json


def validate_json(config_path):
    """
    Validates and parses JSON files in the config folder.
    
    Parameters:
        config_path (str): Path to the folder containing JSON files from mmWave Studio.
    
    Returns:
        tuple: A tuple containing the parsed parameters (Params) and the validation status (jsonValid).
    """

    # Get file path for the JSON files
    filenames = [file for file in os.listdir(config_path) if file.endswith('.json')]

    # Load the JSON files
    if len(filenames) != 2:
        raise ValueError("Too many or missing JSON files in the data folder")

    # Parse the JSON files
    setup_json = None
    mmwave_json = None
    for filename in filenames:
        file_path = os.path.join(config_path, filename)
        print(file_path)
        if 'setup.json' in filename:
            with open(file_path, 'r') as file:
                setup_json = json.load(file)
        elif 'mmwave.json' in filename:
            with open(file_path, 'r') as file:
                mmwave_json = json.load(file)

    if setup_json is None or mmwave_json is None:
        raise ValueError("Required JSON files (setup.json and mmwave.json) are missing or misnamed")

    # Check the validity of the JSON files
    json_valid = checkout_json(setup_json, mmwave_json)

    # Parse the JSON files
    params = parse_json(mmwave_json)
    params['mmWaveDevice'] = setup_json.get('mmWaveDevice')

    return params, json_valid


def checkout_json(setup_json, mmwave_json):
    """
    Validates the setupJSON and mmwaveJSON configurations.

    Parameters:
        setup_json (dict): The parsed setup JSON object.
        mmwave_json (dict): The parsed mmWave JSON object.

    Returns:
        bool: True if the JSON configurations are valid, False otherwise.
    """

    # Supported platform list
    supported_platforms = ["awr1843", "iwr1843", "iwr6843"]
    json_valid = True

    # Validate if the device is supported
    mmwave_device = setup_json.get("mmWaveDevice")
    if mmwave_device not in supported_platforms:
        print(f"Platform not supported: {mmwave_device}")
        json_valid = False

    # Validate the capture hardware
    if setup_json.get("captureHardware") != "DCA1000":
        print(f"Capture hardware is not supported: {setup_json.get('captureHardware')}")
        json_valid = False

    # Validate ADC_ONLY capture
    transfer_fmt_pkt0 = mmwave_json.get("mmWaveDevices", {})[0].get("rawDataCaptureConfig", {}).get("rlDevDataPathCfg_t", {}).get("transferFmtPkt0")
    if transfer_fmt_pkt0 != "0x1":
        print(f"Capture data format is not supported: {transfer_fmt_pkt0}")
        json_valid = False

    # Validate the data logging mode
    data_logging_mode = setup_json.get("DCA1000Config", {}).get("dataLoggingMode")
    if data_logging_mode != "raw":
        print(f"Capture data logging mode is not supported: {data_logging_mode}")
        json_valid = False

    # Validate the capture configuration
    lane_enable = mmwave_json.get("mmWaveDevices", {})[0].get("rawDataCaptureConfig", {}).get("rlDevLaneEnable_t", {}).get("laneEn")
    num_lane = bin(int(lane_enable, 16)).count("1") if lane_enable else 0
    ch_interleave = mmwave_json.get("mmWaveDevices", {})[0].get("rawDataCaptureConfig", {}).get("rlDevDataFmtCfg_t", {}).get("chInterleave")

    if mmwave_device in {"awr1443", "iwr1443", "awr1243", "iwr1243"}:
        if num_lane != 4:
            print(f"{num_lane} LVDS Lane is not supported for device: {mmwave_device}")
            json_valid = False
        if ch_interleave != 0:
            print(f"Interleave mode {ch_interleave} is not supported for device: {mmwave_device}")
            json_valid = False
    else:
        if num_lane != 2:
            print(f"{num_lane} LVDS Lane is not supported for device: {mmwave_device}")
            json_valid = False
        if ch_interleave != 1:
            print(f"Interleave mode {ch_interleave} is not supported for device: {mmwave_device}")
            json_valid = False

    return json_valid


def parse_json(mmwave_json):
    """
    Configures the mmWave devices based on the given JSON data, including enabling channels, setting up profiles, and chirp configurations.
    
    Parameters:
        mmwave_json (dict): The parsed mmWave JSON object.
    
    Returns:
        dict: A dictionary containing the configured parameters for each mmWave device.
    """
    Params = {}
    MmWaveDevicesConfig = mmwave_json["mmWaveDevices"]

    # Number of devices
    Params["NumDevices"] = len(MmWaveDevicesConfig)

    # TX/RX Channels enabled
    TxToEnable = []
    RxToEnable = []
    DevIdMap = []
    Params["DevConfig"] = {}

    for count, deviceConfig in enumerate(MmWaveDevicesConfig):
        DevIdMap.append(deviceConfig["mmWaveDeviceId"] + 1)
        DevId = DevIdMap[count]

        # Initialize device configuration
        Params["DevConfig"][DevId] = {}

        # TX Channel Enable
        TxChannelEn = int(deviceConfig["rfConfig"]["rlChanCfg_t"]["txChannelEn"][2:], 16)
        for txChannel in range(1, 4):
            if TxChannelEn & (1 << (txChannel - 1)):
                TxToEnable.append((3 * (DevId - 1)) + txChannel)

        # RX Channel Enable
        RxChannelEn = int(deviceConfig["rfConfig"]["rlChanCfg_t"]["rxChannelEn"][2:], 16)
        for rxChannel in range(1, 5):
            if RxChannelEn & (1 << (rxChannel - 1)):
                RxToEnable.append((4 * (DevId - 1)) + rxChannel)

        # Profile Configuration
        Params.setdefault('DevConfig', {})[DevId] = {}
        Params['DevConfig'][DevId]['NumProfiles'] = len(MmWaveDevicesConfig[count]['rfConfig']['rlProfiles'])
        NumChirpBlocks = len(MmWaveDevicesConfig[count]['rfConfig']['rlChirps'])
        Profiles = MmWaveDevicesConfig[count]['rfConfig']['rlProfiles']
        Chirps = MmWaveDevicesConfig[count]['rfConfig']['rlChirps']

        for ProfileCount in range(Params['DevConfig'][DevId]['NumProfiles']):
            profile = Profiles[ProfileCount]['rlProfileCfg_t']
            Params['DevConfig'][DevId].setdefault('Profile', {})[ProfileCount] = {
                'ProfileId': profile['profileId'],
                'StartFreq': profile['startFreqConst_GHz'],
                'FreqSlope': profile['freqSlopeConst_MHz_usec'],
                'IdleTime': profile['idleTimeConst_usec'],
                'AdcStartTime': profile['adcStartTimeConst_usec'],
                'RampEndTime': profile['rampEndTime_usec'],
                'TxStartTime': profile['txStartTime_usec'],
                'RxGain': int(profile['rxGain_dB'][2:], 16),
                'NumSamples': profile['numAdcSamples'],
                'SamplingRate': profile['digOutSampleRate'],
                'HpfCornerFreq1': profile['hpfCornerFreq1'],
                'HpfCornerFreq2': profile['hpfCornerFreq2'],
                'Tx0PhaseShift': (int(profile['txPhaseShifter'][2:], 16) & 255) / 4 * 5.625,
                'Tx1PhaseShift': ((int(profile['txPhaseShifter'][2:], 16) >> 8) & 255) / 4 * 5.625,
                'Tx2PhaseShift': ((int(profile['txPhaseShifter'][2:], 16) >> 16) & 255) / 4 * 5.625,
                'Tx0OutPowerBackOff': int(profile['txOutPowerBackoffCode'][2:], 16) & 255,
                'Tx1OutPowerBackOff': (int(profile['txOutPowerBackoffCode'][2:], 16) >> 8) & 255,
                'Tx2OutPowerBackOff': (int(profile['txOutPowerBackoffCode'][2:], 16) >> 16) & 255
            }

        # Chirp Configuration
        Params['DevConfig'][DevId]['NumChirps'] = 0
        for ChirpCount in range(NumChirpBlocks):
            Params['DevConfig'][DevId]['NumChirps'] += (Chirps[ChirpCount]['rlChirpCfg_t']['chirpEndIdx'] - Chirps[ChirpCount]['rlChirpCfg_t']['chirpStartIdx']) + 1
            for ChirpId in range(Chirps[ChirpCount]['rlChirpCfg_t']['chirpStartIdx'] + 1, Chirps[ChirpCount]['rlChirpCfg_t']['chirpEndIdx'] + 2):
                chirp = Chirps[ChirpCount]['rlChirpCfg_t']
                Params['DevConfig'][DevId].setdefault('Chirp', {})[ChirpId] = {
                    'ChirpIdx': ChirpId,
                    'ProfileId': chirp['profileId'],
                    'StartFreqVar': chirp['startFreqVar_MHz'],
                    'FreqSlopeVar': chirp['freqSlopeVar_KHz_usec'],
                    'IdleTimeVar': chirp['idleTimeVar_usec'],
                    'AdcStartTime': chirp['adcStartTimeVar_usec'],
                    'Tx0Enable': bool(int(chirp['txEnable'][2:], 16) & 1),
                    'Tx1Enable': bool(int(chirp['txEnable'][2:], 16) & 2),
                    'Tx2Enable': bool(int(chirp['txEnable'][2:], 16) & 4)
                }

        # Configure the burst profile mappings (BPM) for the device
        NumBpmBlocks = len(MmWaveDevicesConfig[count]['rfConfig']['rlBpmChirps'])
        BpmConfig = MmWaveDevicesConfig[count]['rfConfig']['rlBpmChirps']
        for BpmCount in range(NumBpmBlocks):
            for ChirpId in range(BpmConfig[BpmCount]['rlBpmChirpCfg_t']['chirpStartIdx'] + 1, BpmConfig[BpmCount]['rlBpmChirpCfg_t']['chirpEndIdx'] + 2):
                bpm = BpmConfig[BpmCount]['rlBpmChirpCfg_t']
                Params['DevConfig'][DevId]['Chirp'][ChirpId].update({
                    'Tx0OffBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 1),
                    'Tx0OnBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 2),
                    'Tx1OffBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 4),
                    'Tx1OnBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 8),
                    'Tx2OffBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 16),
                    'Tx2OnBpmVal': bool(int(bpm['constBpmVal'][2:], 16) & 32)
                })

        # Phase shift configuration
        NumPhaseShiftBlocks = len(MmWaveDevicesConfig[count]['rfConfig']['rlRfPhaseShiftCfgs'])
        PhaseShiftConfig = MmWaveDevicesConfig[count]['rfConfig']['rlRfPhaseShiftCfgs']
        for PhaseShiftCount in range(NumPhaseShiftBlocks):
            for ChirpId in range(PhaseShiftConfig[PhaseShiftCount]['rlRfPhaseShiftCfg_t']['chirpStartIdx'] + 1, PhaseShiftConfig[PhaseShiftCount]['rlRfPhaseShiftCfg_t']['chirpEndIdx'] + 2):
                phase_shift = PhaseShiftConfig[PhaseShiftCount]['rlRfPhaseShiftCfg_t']
                Params['DevConfig'][DevId]['Chirp'][ChirpId].update({
                    'Tx0PhaseShift': phase_shift['tx0PhaseShift'] * 5.625,
                    'Tx1PhaseShift': phase_shift['tx1PhaseShift'] * 5.625,
                    'Tx2PhaseShift': phase_shift['tx2PhaseShift'] * 5.625
                })
        
        # Data format configuration
        AdcFormatConfig = MmWaveDevicesConfig[count]['rfConfig']['rlAdcOutCfg_t']['fmt']
        Params['DevConfig'][DevId]['DataFormat'] = {
            'NumAdcBits': AdcFormatConfig['b2AdcBits'],
            'Format': AdcFormatConfig['b2AdcOutFmt'],
            'ReductionFactor': AdcFormatConfig['b8FullScaleReducFctr'],
            'IQSwap': MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['iqSwapSel'],
            'chInterleave': MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevDataFmtCfg_t']['chInterleave'],
            'numLane': bin(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLaneEnable_t']['laneEn'][2:], 16)).count('1')
        }

        if Params["DevConfig"][DevId]["DataFormat"]["NumAdcBits"] == 2:
            if Params["DevConfig"][DevId]["DataFormat"]["Format"] == 0:
                gAdcOneSampleSize = 2  # real data, one sample is 16bits = 2 bytes
            elif Params["DevConfig"][DevId]["DataFormat"]["Format"] in [1, 2]:
                gAdcOneSampleSize = 4  # complex data, one sample is 32bits = 4 bytes
            else:
                print("Error: unsupported ADC dataFmt")
        else:
            print(f"Error: unsupported ADC bits ({Params['DevConfig'][DevId]['DataFormat']['NumAdcBits']})")
        Params["DevConfig"][DevId]["DataFormat"]["gAdcOneSampleSize"] = gAdcOneSampleSize

        # Data path configuration
        DataPathConfig = MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevDataPathCfg_t']
        Params['DevConfig'][DevId]['DataPath'] = {
            'Interface': DataPathConfig['intfSel'],
            'Packet0': int(DataPathConfig['transferFmtPkt0'][2:], 16),
            'Packet1': int(DataPathConfig['transferFmtPkt1'][2:], 16),
            'CqConfig': DataPathConfig['cqConfig'],
            'Cq0TransSize': DataPathConfig['cq0TransSize'],
            'Cq1TransSize': DataPathConfig['cq1TransSize'],
            'Cq2TransSize': DataPathConfig['cq2TransSize']
        }

        if Params['DevConfig'][DevId]['DataPath']['Interface'] == 1:
            Params['DevConfig'][DevId]['DataPath'].update({
                'LaneMap': int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLaneEnable_t']['laneEn'][2:], 16),
                'LvdsFormat': int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneFmtMap']),
                'LvdsMsbFirst': bool(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneParamCfg'][2:], 16) & 1),
                'LvdsMsbCrcPresent': bool(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneParamCfg'][2:], 16) & 2),
                'LvdsPktEndPulse': bool(int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevLvdsLaneCfg_t']['laneParamCfg'][2:], 16) & 4)
            })
        
        if Params['DevConfig'][DevId]['DataPath']['Interface'] == 0:
            Params['DevConfig'][DevId]['DataPath'].update({
                'CsiLane0Pos': int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) & 15,
                'CsiLane1Pos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 4) & 15,
                'CsiLane2Pos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 8) & 15,
                'CsiLane3Pos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 12) & 15,
                'CsiLaneClkPos': (int(MmWaveDevicesConfig[count]['rawDataCaptureConfig']['rlDevCsi2Cfg_t']['lanePosPolSel'][2:], 16) >> 16) & 15
            })

        # Frame configuration
        Params['FrameType'] = 0
        if Params['FrameType'] == 0:
            FrameConfig = MmWaveDevicesConfig[count]['rfConfig']['rlFrameCfg_t']
            Params['DevConfig'][DevId]['FrameConfig'] = {
                'ChirpIdx': FrameConfig['chirpStartIdx'],
                'ChirpEndIdx': FrameConfig['chirpEndIdx'],
                'NumChirpLoops': FrameConfig['numLoops'],
                'NumFrames': FrameConfig['numFrames'],
                'Periodicity': FrameConfig['framePeriodicity_msec'],
                'FrameTriggerDelay': FrameConfig['frameTriggerDelay']
            }
        
        if Params['FrameType'] == 1:
            # Advanced frame configuration
            AdvancedFrameSequence = MmWaveDevicesConfig[count]['rfConfig']['rlAdvFrameCfg_t']['frameSeq']
            AdvancedFrameData = MmWaveDevicesConfig[count]['rfConfig']['rlAdvFrameCfg_t']['frameData']
            Params['DevConfig'][DevId]['AdvFrame'] = {
                'NumFrames': AdvancedFrameSequence['numFrames'],
                'NumSubFrames': AdvancedFrameSequence['numOfSubFrames'],
                'FrameTriggerDelay': AdvancedFrameSequence['frameTrigDelay_usec']
            }
            # Subframe configuration
            for SubFrameId in range(len(AdvancedFrameSequence['subFrameCfg'])):
                sub_frame = AdvancedFrameSequence['subFrameCfg'][SubFrameId]['rlSubFrameCfg_t']
                Params['DevConfig'][DevId]['AdvFrame'].setdefault('SubFrame', {})[SubFrameId] = {
                    'ForceProfileIdx': sub_frame['forceProfileIdx'],
                    'ChirpStartIdx': sub_frame['chirpStartIdx'],
                    'NumChirp': sub_frame['numOfChirps'],
                    'NumChirpLoops': sub_frame['numLoops'],
                    'BurstPeriod': sub_frame['burstPeriodicity_msec'],
                    'ChirpStartIdOffset': sub_frame['chirpStartIdxOffset'],
                    'NumBurst': sub_frame['numOfBurst'],
                    'NumBurstLoops': sub_frame['numOfBurstLoops'],
                    'SubFramePeriod': sub_frame['subFramePeriodicity_msec'],
                    'ChirpsPerDataPkt': AdvancedFrameData['subframeDataCfg'][SubFrameId]['rlSubFrameDataCfg_t']['numChirpsInDataPacket']
                }
        
        if Params['FrameType'] == 2:
            # Continuous mode configuration
            ContinuousModeConfig = MmWaveDevicesConfig[count]['rfConfig']['rlContModeCfg_t']
            Params['DevConfig'][DevId]['ContFrame'] = {
                'StartFreq': ContinuousModeConfig['startFreqConst_GHz'],
                'SamplingRate': ContinuousModeConfig['digOutSampleRate'],
                'Tx0OutPowerBackoffCode': int(ContinuousModeConfig['txOutPowerBackoffCode'][2:], 16) & 255,
                'Tx1OutPowerBackoffCode': (int(ContinuousModeConfig['txOutPowerBackoffCode'][2:], 16) >> 8) & 255,
                'Tx2OutPowerBackoffCode': (int(ContinuousModeConfig['txOutPowerBackoffCode'][2:], 16) >> 16) & 255,
                'Tx0PhaseShifter': int(ContinuousModeConfig['txPhaseShifter'][2:], 16) & 255,
                'Tx1PhaseShifter': (int(ContinuousModeConfig['txPhaseShifter'][2:], 16) >> 8) & 255,
                'Tx2PhaseShifter': (int(ContinuousModeConfig['txPhaseShifter'][2:], 16) >> 16) & 255,
                'RxGain': ContinuousModeConfig['rxGain_dB'],
                'HpfCornerFreq1': ContinuousModeConfig['hpfCornerFreq1'],
                'HpfCornerFreq2': ContinuousModeConfig['hpfCornerFreq2']
            }
    Params['TxToEnable'] = TxToEnable
    Params['RxToEnable'] = RxToEnable
        
    # Return the configuration parameters
    return Params


if __name__ == "__main__":
    
    # Validate the JSON files
    with open("data2parse.yaml", "r") as file:
        data = yaml.safe_load(file)
    config_path = os.path.join("rawData/configs", data["config"])
    params, json_valid = validate_json(config_path)

    if json_valid:
        print("JSON files are valid.")
        print(json.dumps(params, indent=4, sort_keys=True))
    else:
        print("JSON files are invalid.")
        print(json.dumps(params, indent=4, sort_keys=True))