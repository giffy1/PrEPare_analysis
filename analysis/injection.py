# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:20:11 2017

@author: snoran
"""

from client import Client
import time
import json
import datetime

# instantiate the client, passing in a valid user ID:
user_id = "91.e6.eb.8d.65.ee.a5.e3.5a.b3"
c = Client(user_id)

CMD_QUIT = ['q', 'quit']
CMD_HELP = ['h', 'help']
CMD_INJECT_PILL_INTAKE_GESTURE = ['p', 'pill']
CMD_SET_SCHEDULE = ['schd', 'schedule']

ARG_TIMESTAMP = ['t', 'timestamp']
ARG_UUID = ['uuid']
ARG_START_DATE = ['s', 'start']
ARG_END_DATE = ['e', 'end']

VALUE_TIMESTAMP_NOW = ['now']

DEFAULT_UUID = '00:00:00:00:00:00'

help_msg = """
This script is for injecting responses into the pill intake detection system.
Type '{}' or '{}' or Ctrl-C to quit. Type '{}' or '{}' to show this help message. 
To inject a pill intake gesture, type '{}' or '{}'. Arguments include 't' and 'uuid.'
""".format(CMD_QUIT[0], CMD_QUIT[1], CMD_HELP[0], CMD_HELP[1], CMD_INJECT_PILL_INTAKE_GESTURE[0], 
           CMD_INJECT_PILL_INTAKE_GESTURE[1])

def get_input():
    print "Enter a command :"
    return raw_input().lower().strip()
    
def get_medication_input():
    print "Enter a medication (ENTER to stop) :"
    return raw_input().strip()
    
def request_input():
    print '\n'+ help_msg + '\n'
    input_ = get_input()
    kwargs = []
    while input_ not in CMD_QUIT:
        args = []
        if ' ' in input_:
            command, arg_str = input_.split(' ', 1)
            args = arg_str.split(' ')
            for arg in args:
                try:
                    key,value = arg.split("=")
                    kwargs.append((key,value))
                except:
                    pass
        else:
            command = input_
        if command in CMD_HELP:
            print help_msg
        elif command in CMD_QUIT:
            break
        elif command in CMD_INJECT_PILL_INTAKE_GESTURE:
            timestamp = int(time.time()) # only care about second precision
            uuid = DEFAULT_UUID
            for k,v in kwargs:
                if k in ARG_TIMESTAMP:
                    if v in VALUE_TIMESTAMP_NOW:
                        timestamp = int(time.time())
                    else:
                        timestamp = int(v)
                elif k in ARG_UUID:
                    uuid = v

            c.send_socket.send(json.dumps({'user_id' : user_id, 'sensor_type' : 'SENSOR_SERVER_MESSAGE', 'message' : 'PILL_INTAKE_GESTURE','data': {'timestamp' : timestamp, 'bottle_uuid' : uuid, 'event' : 'PILL_INTAKE_GESTURE'}}) + "\n")
            
        elif command in CMD_SET_SCHEDULE:
            # schd s=6/26/2017 e=7/3/2017
            start_date = datetime.date.today()
            end_date = datetime.date.today()
            for k,v in kwargs:
                if k in ARG_START_DATE:
                    start_date=datetime.datetime.strptime(v, "%m/%d/%Y").date()
                elif k in ARG_END_DATE:
                    end_date=datetime.datetime.strptime(v, "%m/%d/%Y").date()
                    
            print start_date,end_date
            
            schedule = {}
            medications = set()
            medication_input = get_medication_input()     
            while medication_input != '':
                if ' ' in medication_input:
                    medication, times =  medication_input.split(' ', 1)
                    print times, "YES"
                    if ' ' in times:
                        schedule[medication] = times.split(' ')
                    else:
                        schedule[medication] = [times, None]
                else:
                    medication = medication_input
                medications.add(medication)
                if len(medications) >= 4:
                    break
                medication_input = get_medication_input()
            
            for medication in medications:
                if medication in schedule:
                    continue
                times = []
                
                print "Enter the morning schedule for {} (leave blank for none): ".format(medication)
                timeToTake = raw_input()
                if timeToTake == '':
                    times.append(None)
                else:
                    times.append(timeToTake)
                    
                print "Enter the afternoon schedule for {} (leave blank for none): ".format(medication)
                timeToTake = raw_input()
                if timeToTake == '':
                    times.append(None)
                else:
                    times.append(timeToTake)
                    
                schedule[medication] = times
            
            adherence_data = {}
            d = start_date
            delta = datetime.timedelta(days=1)
            while d <= end_date:
                adherence_data[str(d)] = {}
                for medication in medications:
                    adherence_type_AM = 'FUTURE'
                    adherence_type_PM = 'FUTURE'
                    if schedule[medication][0] == None:
                        adherence_type_AM = 'NONE'
                    if schedule[medication][1] == None:
                        adherence_type_PM = 'NONE'
                    adherence_data[str(d)][medication] = [(adherence_type_AM, schedule[medication][0]), (adherence_type_PM, schedule[medication][1])]
                d += delta
            
            print adherence_data
            print schedule
                                    
#            c.send_socket.send(json.dumps({'user_id' : user_id, 'sensor_type' : 'SENSOR_SERVER_MESSAGE', 'message' : 'UPDATE_DATA','data': {'start_date' : str(start_date), 'end_date' : str(end_date), 'adherence_data' : adherence_data, 'schedule' : schedule, 'medications' : list(medications)}}) + "\n")
            
        input_ = get_input()
        
    print "Qutting..."
    c.disconnect()
    
c.set_connection_callback(request_input)

# connect to the server to begin:
c.connect()