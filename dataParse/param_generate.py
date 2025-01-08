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
    # params = parse_json(mmwave_json)
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
    num_lane = bin(int(lane_enable, 16)).count("1")if lane_enable else 0
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
        profiles = deviceConfig["rfConfig"]["rlProfiles"]
        Params["DevConfig"][DevId]["NumProfiles"] = len(profiles)
        Params["DevConfig"][DevId]["Profile"] = []

        for profile in profiles:
            profile_cfg = profile["rlProfileCfg_t"]
            profile_dict = {
                "ProfileId": profile_cfg["profileId"],
                "StartFreq": profile_cfg["startFreqConst_GHz"],
                "FreqSlope": profile_cfg["freqSlopeConst_MHz_usec"],
                "IdleTime": profile_cfg["idleTimeConst_usec"],
                "AdcStartTime": profile_cfg["adcStartTimeConst_usec"],
                "RampEndTime": profile_cfg["rampEndTime_usec"],
                "TxStartTime": profile_cfg["txStartTime_usec"],
                "RxGain": int(profile_cfg["rxGain_dB"][2:], 16),
                "NumSamples": profile_cfg["numAdcSamples"],
                "SamplingRate": profile_cfg["digOutSampleRate"],
                "HpfCornerFreq1": profile_cfg["hpfCornerFreq1"],
                "HpfCornerFreq2": profile_cfg["hpfCornerFreq2"],
                "Tx0PhaseShift": (int(profile_cfg["txPhaseShifter"][2:], 16) & 255) / 4 * 5.625,
                "Tx1PhaseShift": ((int(profile_cfg["txPhaseShifter"][2:], 16) >> 8) & 255) / 4 * 5.625,
                "Tx2PhaseShift": ((int(profile_cfg["txPhaseShifter"][2:], 16) >> 16) & 255) / 4 * 5.625,
                "Tx0OutPowerBackOff": int(profile_cfg["txOutPowerBackoffCode"][2:], 16) & 255,
                "Tx1OutPowerBackOff": (int(profile_cfg["txOutPowerBackoffCode"][2:], 16) >> 8) & 255,
                "Tx2OutPowerBackOff": (int(profile_cfg["txOutPowerBackoffCode"][2:], 16) >> 16) & 255,
            }
            Params["DevConfig"][DevId]["Profile"].append(profile_dict)

        # Chirp Configuration
        chirps = deviceConfig["rfConfig"]["rlChirps"]
        Params["DevConfig"][DevId]["NumChirps"] = 0
        Params["DevConfig"][DevId]["Chirp"] = {}

        for chirp in chirps:
            chirp_cfg = chirp["rlChirpCfg_t"]
            start_idx = chirp_cfg["chirpStartIdx"] + 1
            end_idx = chirp_cfg["chirpEndIdx"] + 1

            for chirp_id in range(start_idx, end_idx + 1):
                Params["DevConfig"][DevId]["Chirp"][chirp_id] = {
                    "ChirpIdx": chirp_id,
                    "ProfileId": chirp_cfg["profileId"],
                    "StartFreqVar": chirp_cfg["startFreqVar_MHz"],
                    "FreqSlopeVar": chirp_cfg["freqSlopeVar_KHz_usec"],
                    "IdleTimeVar": chirp_cfg["idleTimeVar_usec"],
                    "AdcStartTime": chirp_cfg["adcStartTimeVar_usec"],
                    "Tx0Enable": int(chirp_cfg["txEnable"][2:], 16) & 1,
                    "Tx1Enable": (int(chirp_cfg["txEnable"][2:], 16) >> 1) & 1,
                    "Tx2Enable": (int(chirp_cfg["txEnable"][2:], 16) >> 2) & 1,
                }

        # Configure the burst profile mappings (BPM) for the device
        NumBpmBlocks = len(deviceConfig["rfConfig"]["rlBpmChirps"])
        BpmConfig = deviceConfig["rfConfig"]["rlBpmChirps"]
        for BpmCount in range(NumBpmBlocks):
            for ChirpId in range(BpmConfig[BpmCount]["rlBpmChirpCfg_t"]["chirpStartIdx"] + 1, BpmConfig[BpmCount]["rlBpmChirpCfg_t"]["chirpEndIdx"] + 1):
                bpm_values = int(BpmConfig[BpmCount]["rlBpmChirpCfg_t"]["constBpmVal"][2:], 16)
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx0OffBpmVal"] = (bpm_values >> 0) & 1
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx0OnBpmVal"] = (bpm_values >> 1) & 1
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx1OffBpmVal"] = (bpm_values >> 2) & 1
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx1OnBpmVal"] = (bpm_values >> 3) & 1
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx2OffBpmVal"] = (bpm_values >> 4) & 1
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx2OnBpmVal"] = (bpm_values >> 5) & 1

        # Phase shift configuration
        NumPhaseShiftBlocks = len(deviceConfig["rfConfig"]["rlRfPhaseShiftCfgs"])
        PhaseShiftConfig = deviceConfig["rfConfig"]["rlRfPhaseShiftCfgs"]
        for PhaseShiftCount in range(NumPhaseShiftBlocks):
            for ChirpId in range(PhaseShiftConfig[PhaseShiftCount]["rlRfPhaseShiftCfg_t"]["chirpStartIdx"] + 1,
                                 PhaseShiftConfig[PhaseShiftCount]["rlRfPhaseShiftCfg_t"]["chirpEndIdx"] + 1):
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx0PhaseShift"] = PhaseShiftConfig[PhaseShiftCount]["rlRfPhaseShiftCfg_t"]["tx0PhaseShift"] * 5.625
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx1PhaseShift"] = PhaseShiftConfig[PhaseShiftCount]["rlRfPhaseShiftCfg_t"]["tx1PhaseShift"] * 5.625
                Params["DevConfig"][DevId]["Chirp"][ChirpId]["Tx2PhaseShift"] = PhaseShiftConfig[PhaseShiftCount]["rlRfPhaseShiftCfg_t"]["tx2PhaseShift"] * 5.625

        # Data format configuration
        AdcFormatConfig = deviceConfig["rfConfig"]["rlAdcOutCfg_t"]["fmt"]
        Params["DevConfig"][DevId]["DataFormat"] = {
            "NumAdcBits": AdcFormatConfig["b2AdcBits"],
            "Format": AdcFormatConfig["b2AdcOutFmt"],
            "ReductionFactor": AdcFormatConfig["b8FullScaleReducFctr"],
            "IQSwap": deviceConfig["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["iqSwapSel"],
            "chInterleave": deviceConfig["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["chInterleave"],
            "numLane": numberOfEnabledChan(int(deviceConfig["rawDataCaptureConfig"]["rlDevLaneEnable_t"]["laneEn"][2:], 16))
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
        DataPathConfig = deviceConfig["rawDataCaptureConfig"]["rlDevDataPathCfg_t"]
        Params["DevConfig"][DevId]["DataPath"] = {
            "Interface": DataPathConfig["intfSel"],
            "Packet0": int(DataPathConfig["transferFmtPkt0"][2:], 16),
            "Packet1": int(DataPathConfig["transferFmtPkt1"][2:], 16),
            "CqConfig": DataPathConfig["cqConfig"],
            "Cq0TransSize": DataPathConfig["cq0TransSize"],
            "Cq1TransSize": DataPathConfig["cq1TransSize"],
            "Cq2TransSize": DataPathConfig["cq2TransSize"]
        }

        if Params["DevConfig"][DevId]["DataPath"]["Interface"] == 1:
            Params["DevConfig"][DevId]["DataPath"].update({
                "LaneMap": int(deviceConfig["rawDataCaptureConfig"]["rlDevLaneEnable_t"]["laneMap"][2:], 16),
                "LaneMapSel": int(deviceConfig["rawDataCaptureConfig"]["rlDevLaneEnable_t"]["laneMapSel"][2:], 16),
                "Frame0": int(deviceConfig["rawDataCaptureConfig"]["rlDevFrame0_t"]["frameSize"][2:], 16),
                "Frame1": int(deviceConfig["rawDataCaptureConfig"]["rlDevFrame1_t"]["frameSize"][2:], 16),
                "Frame2": int(deviceConfig["rawDataCaptureConfig"]["rlDevFrame2_t"]["frameSize"][2:], 16),
                "RxChannel0": int(deviceConfig["rawDataCaptureConfig"]["rlDevDataPathCfg_t"]["rx0ChannelEnable"][2:], 16),
                "RxChannel1": int(deviceConfig["rawDataCaptureConfig"]["rlDevDataPathCfg_t"]["rx1ChannelEnable"][2:], 16)
            })

        # Frame configuration
        FrameConfig = deviceConfig["frameConfig"]
        Params["DevConfig"][DevId]["FrameConfig"] = {
            "FrameRate": FrameConfig["frameRate"],
            "NumFrames": FrameConfig["numFrames"],
            "StartIdx": FrameConfig["frameStartIdx"],
            "EndIdx": FrameConfig["frameEndIdx"]
        }

    # Return the configuration parameters
    return Params


        # Additional Configurations (BPM, Phase Shift, Data Format, Data Path) can be added similarly...

    return Params



if __name__ == "__main__":
    
    # Validate the JSON files
    with open("data2parse.yaml", "r") as file:
        data = yaml.safe_load(file)
    config_path = os.path.join("rawData/configs", data["config"])
    params, json_valid = validate_json(config_path)

    if json_valid:
        print("JSON files are valid.")
        print(params)
    else:
        print("JSON files are invalid.")
        print(params)