import numpy as np
import struct
from dataclasses import dataclass
from typing import Union
import matplotlib.pyplot as plt

@dataclass
class Parameter:
    inclass: bytes
    unit_Scaler: int
    cnits: list
    value: Union[int, np.complex128, str, float]
    name: str

@dataclass
class Unit_Struct:
    prefix: str
    unit: str
    power: int

class JEOL_parser:

    data_formats = [
        "INVALID",
        "One_D",
        "Two_D",
        "Three_D",
        "Four_D",
        "Five_D",
        "Six_D",
        "Seven_D",
        "Eight_D",
        "INVALID",
        "INVALID",
        "INVALID",
        "Small_Two_D",
        "Small_Three_D",
        "Small_Four_D"
    ]

    instruments = [
        "NONE",
        "GSX",
        "ALPHA",
        "ECLIPSE",
        "MASS_SPEC",
        "COMPILER",
        "OTHER_NMR",
        "UNKNOWN",
        "GEMINI",
        "UNITY",
        "ASPECT",
        "UX",
        "FELIX",
        "LAMBDA",
        "GE_1280",
        "GE_OMEGA",
        "CHEMAGNETICS",
        "CDFF",
        "GALACTIC",
        "TRIAD",
        "GENERIC_NMR",
        "GAMMA",
        "JCAMP_DX",
        "AMX",
        "DMX",
        "ECA",
        "ALICE",
        "NMR_PIPE",
        "SIMPSON"
    ]    

    prefixes = [    
        "None",
        "Milli",
        "Micro",
        "Nano",
        "Pico",
        "Femto",
        "Atto",
        "Zepto",
        "INVALID",
        "Yotta",
        "Zetta",
        "Exa",
        "Pecta",
        "Giga",
        "Mega",
        "Kilo"
    ]

    units = [
        "None",
        "Abundance",
        "Ampere",
        "Candela",
        "Celsius",
        "Coulomb",
        "Degree",
        "Electronvolt",
        "Farad",
        "Sievert",
        "Gram",
        "Gray",
        "Henry",
        "Hertz",
        "Kelvin",
        "Joule",
        "Liter",
        "Lumen",
        "Lux",
        "Meter",
        "Mole",
        "Newton",
        "Ohm",
        "Pascal",
        "Percent",
        "Point",
        "Ppm",
        "Radian",
        "Second",
        "Siemens",
        "Steradian",
        "Tesla",
        "Volt",
        "Watt",
        "Weber",
        "Decibel",
        "Dalton",
        "Thompson",
        "Ugeneric",
        "LPercent",
        "PPT",
        "PPB",
        "Index"
    ]

    axis_type = [
        "None",
        "Real",
        "TPPI",
        "Complex",
        "Real_Complex",
        "Envelope"
    ]
    
    def __init__(fileName):
        JEOL_parser.fileName = fileName

    def unit_from_bytes(bt):
        return Unit_Struct(JEOL_parser.prefixes[bt[0] >> 4], JEOL_parser.units[bt[1]], bt[0] & 0xF)

    def get_header(mfile):
        header = {}
        with open(mfile, mode='rb') as file:
            fC = file.read()
            #parse the header section
            header["File_Identifier"] = struct.unpack('8s', fC[:8])[0].rstrip(b'\x00') .decode('utf-8')
            header["Endian"] = "Little" if fC[9] else "Big"
            header["Major_Version"] = "1" if fC[9] else "0"
            header["Minor_Version"] = struct.unpack('>H', fC[10:12])[0]
            header["Data_Dimension_Number"] = int(fC[12])
            header["Data_Dimension_Exist"] = bin(fC[13])
            header["Data_Type"] = "32Bit Float" if ((fC[14] & 0b1100000)>>6) else "64Bit Float"
            header["Data_Format"] = JEOL_parser.data_formats[int(fC[14] & 0b00111111)]
            header["Instrument"] = JEOL_parser.instruments[int(fC[15])]
            header["Translate"] = [i for i in fC[16:24]]
            header["Data_Axis_Type"] = [JEOL_parser.axis_type[i] for i in fC[24:32]]
            header["Data_Units"] = [JEOL_parser.unit_from_bytes(fC[2*pos + 32:2*pos + 34]) for pos in range(8)]
            header["Title"] = struct.unpack('124s', fC[48:172])[0].rstrip(b'\x00').decode('utf-8')
            header["Data_Axis_Ranged"] = np.concatenate([[fC[172+i] >> 4, fC[172+i] & 4] for i in range(4)])
            header["Data_Points"] = np.array(struct.unpack('>IIIIIIII', fC[176:208]))
            header["Data_Offset_Start"] = struct.unpack('>IIIIIIII', fC[208:240])
            header["Data_Offset_Stop"] = struct.unpack('>IIIIIIII', fC[240:272])
            header["Data_Axis_Start"] = struct.unpack('>dddddddd', fC[272:336])
            header["Data_Axis_Stop"] = struct.unpack('>dddddddd', fC[336:400])
            header["Creation_Time"] = fC[400:404]
            header["Revision_Time"] = fC[404:408]
            header["Node_Name"] = struct.unpack('16s', fC[408:424])[0].rstrip(b'\x00').decode('utf-8')
            header["Site"] = struct.unpack('128s', fC[424:552])[0].rstrip(b'\x00').decode('utf-8')
            header["Author"] = struct.unpack('128s', fC[552:680])[0].rstrip(b'\x00').decode('utf-8')
            header["Comment"] = struct.unpack('128s', fC[680:808])[0].rstrip(b'\x00').decode('utf-8')
            header["Data_Axis_Titles"] = [struct.unpack('32s', fC[808+i*8:808+i*8+32])[0].rstrip(b'\x00').decode('utf-8') 
                for i in range(8)]
            header["Base_Freq"] = struct.unpack('>dddddddd', fC[1064:1128])
            header["Zero_Point"] = struct.unpack('>dddddddd', fC[1128:1192])
            header["Reversed"] = struct.unpack('????????', fC[1192:1200])
            header["Annotation_Ok"] = fC[1203] >> 7
            header["History_Used"] = struct.unpack('>I', fC[1204:1208])[0]
            header["History_Length"] = struct.unpack('>I', fC[1208:1212])[0]
            header["Param_Start"] = struct.unpack('>I', fC[1212:1216])[0]
            header["Param_Length"] = struct.unpack('>I', fC[1216:1220])[0]
            header["List_Start"] = struct.unpack('>IIIIIIII', fC[1220:1252])
            header["List_Length"] = struct.unpack('>IIIIIIII', fC[1252:1284])
            header["Data_Start"] = struct.unpack('>I', fC[1284:1288])[0]
            header["Data_Length"] = struct.unpack('>Q', fC[1288:1296])[0]
            header["Context_Start"] = struct.unpack('>Q', fC[1296:1304])[0]
            header["Context_Length"] = struct.unpack('>I', fC[1304:1308])[0]
            header["Annote_Start"] = struct.unpack('>Q', fC[1308:1316])[0]
            header["Annote_Length"] = struct.unpack('>I', fC[1316:1320])[0]
            header["Total_Size"] = struct.unpack('>Q', fC[1320:1328])[0]
            header["Unit_Location"] = struct.unpack('????????', fC[1328:1336])
            header["Compound_Units"] = (struct.unpack('>h', fC[1336:1338])[0], [JEOL_parser.unit_from_bytes(fC[2*pos + 1338:2*pos + 1340])  for pos in range(5)],
            struct.unpack('>h', fC[1348:1350]), [JEOL_parser.unit_from_bytes(fC[2*pos + 1350:2*pos + 1352]) for pos in range(5)])

        return header

    def parse_param(pdata):
        value_type = pdata[32]
        value = 0
        if(value_type == 0):
            value = struct.unpack('16s', pdata[16:32])[0].rstrip(b'\x00').decode('utf-8')
        elif(value_type == 1):
            value = struct.unpack('I', pdata[16:20])[0]
        elif(value_type == 2):
            value = struct.unpack('d', pdata[16:24])[0]
        elif(value_type == 3):
            value = struct.unpack('Q', pdata[16:24])[0] + struct.unpack('Q', pdata[16:24])[0] * 1j
        elif(value_type == 4):
            inf_decode = struct.unpack('I', pdata[16:20])[0]
            if(inf_decode == 1): value = -np.inf
            elif(inf_decode == 2): value = -1
            elif(inf_decode == 3): value = 0
            elif(inf_decode == 4): value = 1
            elif(inf_decode == 5): value = np.inf

        tparam = Parameter(
            pdata[0:4],
            struct.unpack('H', pdata[4:6])[0],
            [JEOL_parser.unit_from_bytes(pdata[6+2*i:8+2*i]) for i in range(5)],
            value,
            struct.unpack('28s', pdata[36:64])[0].rstrip(b'\x00').rstrip(b' ').decode('utf-8')
        )

        return tparam

    def get_params(mfile):

        params = {}    
        param_header = {}
        with open(mfile, mode='rb') as file:
            fC = file.read()

            #Get the start position of the parameter header
            pstart = struct.unpack('>I', fC[1212:1216])[0]

            #Parse the parameter header. Endian-ness seems to flip here, which is irritating
            param_header["Parameter_Size"] = struct.unpack('I', fC[pstart:pstart+4])[0]
            param_header["Low_Index"] = struct.unpack('I', fC[pstart+4:pstart+8])[0]
            param_header["High_Index"] = struct.unpack('I', fC[pstart+8:pstart+12])[0]
            param_header["Total_Size"] = struct.unpack('I', fC[pstart+12:pstart+16])[0]

            for i in range(param_header["High_Index"]):
                param = JEOL_parser.parse_param(fC[pstart+16+param_header["Parameter_Size"]*i:pstart+16+param_header["Parameter_Size"]*(i+1)])
                params[param.name] = param
            
            return (param_header, params)

    def get_1d_data(mfile):
        hdr = JEOL_parser.get_header(mfile)

        #check that I can actually do this
        if(hdr["Data_Axis_Type"][0] != "Complex" or hdr["Data_Format"] != "One_D"):
            return -1

        dlen = hdr["Data_Points"][0]
        r_dstart = hdr["Data_Start"]
        i_dstart = r_dstart + dlen*8

        data_arr = np.zeros(dlen, dtype='complex128')
        with open(mfile, mode='rb') as file:
            fC = file.read()

            for i in range(dlen):
                data_arr[i] = (struct.unpack('d', fC[r_dstart+i*8:r_dstart+i*8+8])[0]
                    + 1j * struct.unpack('d', fC[i_dstart+i*8:i_dstart+i*8+8])[0])
        
        return data_arr

    def get_ruler(mfile):
        hdr = JEOL_parser.get_header(mfile)

        #check that I can actually do this
        if(hdr["Data_Axis_Type"][0] != "Complex" or hdr["Data_Format"] != "One_D"):
            return -1

        #check that the data format is ranged
        if(hdr["Data_Axis_Ranged"][0] != 0):
            return -1

        dlen = hdr["Data_Points"][0]
        start = hdr["Data_Axis_Start"][0]
        stop = hdr["Data_Axis_Stop"][0]
        return np.arange(start, stop, (stop-start)/dlen)