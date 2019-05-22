#
# TI SimpleLink SensorTag 2015
# Date: 2015 07 07
#
# Sensor: Gyro, Accelerometer and Magnetometer
# Values: 9 bytes x, y, z for each sensor
# Note: Sensor values have not been validated
#
import struct, sys, traceback, time, threading, pprint, csv
from bluepy.btle import UUID, Peripheral, BTLEException
from datetime import datetime

def TI_UUID(val):
    return UUID("%08X-0451-4000-b000-000000000000" % (0xF0000000+val))

config_uuid = TI_UUID(0xAA82)
data_uuid = TI_UUID(0xAA81)
period_uuid = TI_UUID(0xAA83)

dataL = {}
dataR = {}
dataC = {}

#sensorOnVal = gyroOn | magOn | accOn
sensorOnVal = 0x7F02

#sensorOn = struct.pack("BB", (sensorOnVal >> 8) & 0xFF, (sensorOnVal & 0xFF))
sensorOn = struct.pack("BB", 0x7F, 0x02)
sensorOff = struct.pack("BB", 0x00, 0x00)

period = struct.pack("B",0x0A)

if len(sys.argv) != 4:
    print ("Fatal, must pass label and time in seconds: <label> <time> <personName>")
    quit()

label = int(sys.argv[1])
time_read = float(sys.argv[2])
personName = sys.argv[3]

sensors_connected = 0
e = threading.Event()

def read_data(sensorName,sensorMAC):
    global sensors_connected
    try:
        print ("Info, trying to connect to %s: %s" % (sensorName,sensorMAC))
        sensor = Peripheral(sensorMAC)

    except BTLEException:
        print ("Fatal, unable to connect!")
  
    except:
        print ("Fatal, unexpected error!")
        traceback.print_exc()
        raise
    else:
        try:
            print ("Info, connected and turning sensor %s on!" % sensorName)
            sh = sensor.getCharacteristics(uuid=config_uuid)[0]
            sh.write(sensorOn, withResponse=True) 

            tLock = threading.Lock()

            with tLock:
                sensors_connected += 1
            e.set()

            while sensors_connected < 3:    
                e.wait()

                

            print ("Info, reading values on %s!" % sensorName)

            sh = sensor.getCharacteristics(uuid=data_uuid)[0]

            t_end = time.time() + time_read

            index = 0
            beginning = time.time()
            while time.time() <= t_end:
                index = index + 1
                rawVals = sh.read()

                #   Movement data: 9 bytes made up of x, y and z for Gyro, Accelerometer, 
                #   and Magnetometer.  Raw values must be divided by scale
                (gyroX, gyroY, gyroZ, accX, accY, accZ, magX, magY, magZ) = struct.unpack('<hhhhhhhhh', rawVals)
                scale = 4096.0

                
                if(sensorName == 'Left'):
                    dataL[index] = [time.time() - beginning, accX / scale, accY / scale, accZ / scale, label]
                elif(sensorName == 'Right'):
                    dataR[index] = [time.time() - beginning, accX / scale, accY / scale, accZ / scale, label]
                else:
                    dataC[index] = [time.time() - beginning, accX / scale, accY / scale, accZ / scale, label]
            
            print ("Info, turning sensor %s off!" % sensorName)
            sh = sensor.getCharacteristics(uuid=config_uuid)[0]
            sh.write(sensorOff, withResponse=True)
            sensors_connected -= 1
        except:
            print ("Fatal, unexpected error!")
            traceback.print_exc()
            raise

        finally:
            print ("Info, disconnecting!")
            sensor.disconnect()
    finally:
        quit()



# Bit settings to turn on individual movement sensors
# bits 0 - 2: Gyro x, y z
# bits 3 - 5: Accelerometer x, y, z
# bit: 6: Magnetometer turns on X, y , z with one bit

#gyroOn = 0x0700
#accOn = 0x3800
#magOn = 0xC001

sensorLeft = '24:71:89:BB:FA:00'
sensorRight = '24:71:89:C0:BC:84'
sensorChest = '24:71:89:C1:3C:04'

lThread = threading.Thread(target=read_data,args=("Left",sensorLeft,))
lThread.start()
rThread = threading.Thread(target=read_data,args=("Right",sensorRight,))
rThread.start()
cThread = threading.Thread(target=read_data,args=("Chest",sensorChest,))
cThread.start()

lThread.join()
print("End of left thread!!")
rThread.join()
print("End of right thread!!")
cThread.join()
print("End of chest thread!!")

file = "DataSets/Still_" + personName 

file_name = file + ".csv"

data_file = open(file_name, mode='w')

data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for indexL in dataL.keys():
    if indexL in dataR.keys() and indexL in dataC.keys():
        timestampL = dataL[indexL][0]
        timestampR = dataR[indexL][0]
        timestampC = dataC[indexL][0]

        # print('%d, %f, %f, %f' % (indexL,timestampL,timestampR,timestampC))
        # avg_timestamp = sum([timestampL,timestampC,timestampR]) / 3
        data_writer.writerow([indexL,dataL[indexL][1], dataL[indexL][2], dataL[indexL][3], \
            dataR[indexL][1], dataR[indexL][2], dataR[indexL][3], dataC[indexL][1], dataC[indexL][2], dataC[indexL][3], dataC[indexL][4]])
            # print("%d,%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%2.2f,%d" % (indexL, dataL[indexL][1], dataL[indexL][2], dataL[indexL][3], \
            #         dataR[indexL][1], dataR[indexL][2], dataR[indexL][3], dataC[indexL][1], dataC[indexL][2], dataC[indexL][3], dataC[indexL][4]))