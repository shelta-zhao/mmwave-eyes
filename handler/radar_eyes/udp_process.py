"""
    Author        : Shelta Zhao(赵小棠)
    Email         : xiaotang_zhao@outlook.com
    Copyright (C) : NJU DisLab, 2025.
    Description   : This script defines the UDP data processor.
"""

import os
import copy
import numpy as np
from tqdm import tqdm


class UdpDataProcessor:

    def __init__(self, udpObj={}):
        """
        Initialize the UDP data processor.

        Parameters:
            udpObj (dict): The object containing the UDP packet data.
        """

        # Initialize the UDP packet object
        self.udpObj = udpObj
        self.seqRatio = np.array([1, 256, 256 << 8, 256 << 16])
        self.dataSizeRatio = np.array([1, 256, 256 << 8, 256 << 16, 256 << 24, 256 << 32])       
    
    def save_radar_ele(self, data_path):
        """
        Parse the radar data and save the parsed data.

        Parameters:
            data_path (str): The path to the radar data.
        """

        # Check the scenario
        if "outdoors" in data_path:
            self.udpObj.update({
                "numChirpsIn1Loop": 1,
                "numLoopsIn1Frame": 1,
                "numChirpsPerFrame": 1,
                "numRxChan": 4,
                "numAdcSamples": 640,
                "framePeriodicty": 0.005,
                "dataSizeOneFrame": 4 * 640 * 4 * 1 * 1,
            })
        else:
            self.udpObj.update({
                "numChirpsIn1Loop": 3,
                "numLoopsIn1Frame": 8,
                "numChirpsPerFrame": 24,
                "numRxChan": 4,
                "numAdcSamples": 128,
                "framePeriodicty": 0.005,
                "dataSizeOneFrame": 4 * 128 * 4 * 3 * 8,
            })

        # Initialize the UDP packet object
        udpPacketObj = {
            'total_data_size': 0,
            'frameDataBuff_index': 0,
            'frameDataBuff': np.empty(1000000000, dtype=np.uint8),
            'prevUdpRawData': 0,
            'oneFrameDataSize': self.udpObj['dataSizeOneFrame'],
            'frameId': 0,
            'zeroPadding': np.zeros(self.udpObj['dataSizeOneFrame'], dtype=np.uint8, order='C')
        }
        
        # Parse the UDP packets
        all_frames, del_frame_index = self.parse_udp(data_path, udpPacketObj)
        
        # Save the parsed data 
        ADC_path = os.path.join(file_path, "1843_ele", "ADC")
        if not os.path.exists(ADC_path):
            os.makedirs(ADC_path)
        np.save(os.path.join(ADC_path, "all_frames"), all_frames)

        # Save the frame shape
        with open(os.path.join(ADC_path, "frame_shape.txt"), "w") as f:
            f.write(f"Shape: {all_frames.shape}")

        # Save the missing frame index
        file_path = os.path.join(data_path, "del_frame.txt")
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, "w") as file:
            for index in del_frame_index:
                file.write(f"{index}\n")      

        # Update the timestamp file
        self.complete_timestamp(os.path.join(data_path, "timestamp.txt"), all_frames.shape[0], time_interval=self.udpObj['framePeriodicty'])

    def parse_udp(self, data_path, udpPacketObj):
        """
        Parse the UDP packets from the given file.

        Parameters:
            data_path (str): The path to the file.
            udpPacketObj (dict): The object containing the UDP packet data.

        Returns:
            np.ndarray: The parsed data.
            list: The list of missing frame index.
        """
        
        # Initialize the basic parameters
        data_name = os.path.join(data_path, "1843_ele", "udpData.dat")
        udp_index, expected_seq_num, del_frame_index, all_frames, fsize = -1, 1, [], [], os.path.getsize(data_name)
        progress_bar = tqdm(total=fsize, desc="Processing UDP Packets", unit="packet", ncols=90)

        # Parse the UDP packets
        while True:
            
            # Load one UDP packet
            flag, udp_index, currUdpPacketData, _ = self.getOneUdpPacket(data_name, udp_index)
            
            # Check the status of the UDP packet
            if flag == 0:
                break
            elif flag == -1:
                continue

            # Extract the data from the current UDP packet
            current_seq_num = np.matmul(currUdpPacketData[0:4], self.seqRatio)

            # Check if the packet is lost
            if current_seq_num != expected_seq_num and (current_seq_num != 1):
    
                # Handle the missing packets with padding
                missing_packets = current_seq_num - expected_seq_num
                for _ in range(missing_packets):  # Fill missing packets + current packet
                    dynamic_padding = copy.deepcopy(currUdpPacketData)
                    dynamic_padding[0:4] = np.array([
                        (expected_seq_num & 0x000000FF),
                        (expected_seq_num & 0x0000FF00) >> 8,
                        (expected_seq_num & 0x00FF0000) >> 16,
                        (expected_seq_num & 0xFF000000) >> 24
                    ], dtype=np.uint8)

                    self.readOneUdpPacket(udpPacketObj, dynamic_padding)
                    self.fetchFramesData(udpPacketObj)
                    expected_seq_num += 1

                # Extract the frames data
                expected_seq_num += 1
                self.readOneUdpPacket(udpPacketObj, currUdpPacketData)
                flag, new_frames, new_index = self.fetchFramesData(udpPacketObj)
                if udpPacketObj["frameId"] not in del_frame_index:
                    del_frame_index.append(udpPacketObj["frameId"])
                
            else:
                self.readOneUdpPacket(udpPacketObj, currUdpPacketData)
                flag, new_frames, new_index = self.fetchFramesData(udpPacketObj)
                expected_seq_num += 1

            if flag == 1:
                # Save the frames data
                for ii in range(new_index.size):
                    frame_data = self.parse_frame(new_frames[ii])
                    all_frames.append(frame_data)
  
            # Update the progress bar
            progress_bar.n = udp_index
            progress_bar.refresh()
        
        # Close the progress bar
        progress_bar.close()

        # Return the parsed data & the missing frame index
        return np.array(all_frames).astype(np.complex64), del_frame_index

    def getOneUdpPacket(self, filename, udp_index):
        """
        Get one UDP packet from the given file.
        
        Parameters:
            filename (str): The path to the file.
            udp_index (int): The index pointing to the current UDP packet.
        
        Returns:
            flag (int): The flag indicating the status of the UDP packet.
            udp_index (int): The index of the next UDP packet.
            currUdpData (np.ndarray): The current UDP packet data.
            prefix (np.ndarray): The prefix of the UDP packet.
        """

        flag, udp_index, currUdpData, fsize = 0, np.int64(udp_index + 1), 0, os.path.getsize(filename)

        # Check if the udp packet is complete
        if udp_index > (fsize - 2000):
            prefix = np.array([], dtype=np.uint8)
            return flag, udp_index, currUdpData, prefix

        # Check if the udp packet format is correct
        startCodeArray = np.fromfile(filename, dtype=np.uint8, count=4, offset=udp_index)
        if np.sum(np.equal(startCodeArray, [3, 2, 8, 0])) == 4:
            pl_Array = np.fromfile(filename, dtype=np.uint8, count=2, offset=udp_index + 4).astype(np.uint16)
            packet_length =pl_Array[0] + pl_Array[1] * 256
            startCodeNextArray = np.fromfile(filename, dtype=np.uint8, count=4, offset=udp_index + packet_length + 6)

            # Load the udp packet data if the next start code is correct
            if np.sum(np.equal(startCodeNextArray, [3, 2, 8, 0])) == 4:
                prefix = np.concatenate((np.array([3, 2, 8, 0], dtype=np.uint8), pl_Array))
                currUdpData = np.fromfile(filename, dtype=np.uint8, count=packet_length, offset=udp_index + 6)
                udp_index += packet_length + 5
                flag = 1
            else:
                flag = -1
        else:
            flag = -1
        
        return flag, udp_index, currUdpData, prefix
    
    def readOneUdpPacket(self, obj, currUdpData):
        """
        Read the data from the current UDP packet.

        Parameters:
            obj (dict): The object containing the UDP packet data.
            currUdpData (np.ndarray): The current UDP packet data.
            first_packet (bool): The flag indicating the first packet.
        
        Returns:
                int: The status of the UDP packet.
        """

        if currUdpData.size == 0:
            return -1

        obj.update({
            'current_seq_num': np.matmul(currUdpData[:4], self.seqRatio),
            'dataSizeTransed': np.matmul(currUdpData[4:10], self.dataSizeRatio)
        })
        if obj['total_data_size'] and not obj['dataSizeTransed']:
            return -1
        obj['total_data_size'] = obj['total_data_size'] + currUdpData.size - 10
        currUdpRawData = currUdpData[10:]

        if  obj["current_seq_num"].item() == 1:
            obj['prevUdpRawData'] = currUdpRawData
        else:
            obj['frameDataBuff'][obj['frameDataBuff_index']: obj['frameDataBuff_index'] + obj['prevUdpRawData'].size] = obj['prevUdpRawData']
            obj['frameDataBuff_index'] += obj['prevUdpRawData'].size
            obj['prevUdpRawData'] = currUdpRawData
        
        return 1

    def fetchFramesData(self, obj):
        """
        Fetch the data in the frames.

        Parameters:
            obj (dict): The object containing the UDP packet data.

        Returns:
            int: The status of the frame data.
            np.ndarray: The data of the frame.
            int: The index of the frame.
        """

        flag, frames_data, frames_index = 0, np.array([]), np.array([])
        frame_num = int(np.fix(obj['frameDataBuff_index'] / obj['oneFrameDataSize']))
        
        if frame_num > 0:   
            flag = 1
            for i in range(frame_num):
                obj['frameId'] = obj['frameId'] + 1
                currFrameData = obj['frameDataBuff'][0:obj['oneFrameDataSize']]
                currFrameData = np.frombuffer(currFrameData.data, dtype=np.int16, count=int(currFrameData.size / 2), offset=0)
                obj['frameDataBuff'] = np.concatenate((obj['frameDataBuff'][obj['oneFrameDataSize']:], obj['zeroPadding']))
                obj['frameDataBuff_index'] = obj['frameDataBuff_index'] - obj['oneFrameDataSize']
                if i == 0:
                    frames_data = currFrameData[None, :]
                else:
                    frames_data = np.concatenate((frames_data, currFrameData[None, :]))
                frames_index = np.concatenate((frames_index, np.array([obj['frameId']])))

        return flag, frames_data, frames_index

    def parse_frame(self, frame_data):
        """
        Parse the frame data.

        Parameters:
            frame_data (np.ndarray): The frame data to be parsed.

        Returns:
            np.ndarray: The parsed frame data
        """

        # Initialize the frame data parameters
        numLoopsIn1Frame, numChirpsIn1Loop = self.udpObj["numLoopsIn1Frame"], self.udpObj["numChirpsIn1Loop"]
        numAdcSamples, numRxChan, numChirpsPerFrame = self.udpObj["numAdcSamples"], self.udpObj["numRxChan"], self.udpObj["numChirpsPerFrame"]
        frameComplex = np.zeros((numChirpsPerFrame, numRxChan, numAdcSamples), dtype=complex)
        frameComplexFinal = np.zeros((numLoopsIn1Frame, numChirpsIn1Loop, numRxChan, numAdcSamples), dtype=complex)

        # Parse the frame data
        rawData4 = np.reshape(frame_data, (4, int(frame_data.size / 4)), order='F')
        rawDataI = np.reshape(rawData4[0:2, :], (-1, 1), order='F')
        rawDataQ = np.reshape(rawData4[2:4, :], (-1, 1), order='F')
        frameCplx = rawDataI + 1j * rawDataQ
        frameCplxTemp = np.reshape(frameCplx, (numAdcSamples * numRxChan, numChirpsPerFrame), order='F')
        frameCplxTemp = np.transpose(frameCplxTemp, (1, 0))

        for jj in range(0, numChirpsPerFrame, 1):
            frameComplex[jj, :, :] = np.transpose(np.reshape(frameCplxTemp[jj, :], (numAdcSamples, numRxChan), order='F'), (1, 0))

        for nLoop in range(0, numLoopsIn1Frame, 1):
            for nChirp in range(0, numChirpsIn1Loop, 1):
                frameComplexFinal[nLoop, nChirp, :, :] = frameComplex[nLoop * numChirpsIn1Loop + nChirp, :, :]
        frameComplexFinalTmp = np.transpose(frameComplexFinal, (3, 0, 2, 1))

        return frameComplexFinalTmp

    def complete_timestamp(self, file_name, frame_num, time_interval):
        """
        Complete the timestamp file with the given time interval.

        Parameters:
            file_name (str): The path to the timestamp file.
            frame_num (int): The number of frames.
            time_interval (float): The time interval between frames.
        """

        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        if len(lines) > 0:
            start_time_line = lines[0].strip()
            start_time = float(start_time_line.split(':')[1].strip())

            # Generate the new timestamps
            new_timestamps = [f"{start_time + time_interval * i:.6f}\n" for i in range(0, frame_num)]
            lines = [lines[0]] + new_timestamps + [lines[-1]]

            # Rewrite the timestamp file
            with open(file_name, 'w', encoding='utf-8') as file:
                file.writelines(lines)
        else:
            print('Skip Warning: timestamp file is empty!')


if __name__ == "__main__":

    processor = UdpDataProcessor()
    processor.save_radar_ele("data/adc_data/2023_06_30_20_13_01")

