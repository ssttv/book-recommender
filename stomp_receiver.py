import time
import sys
from datetime import datetime
from threading import Lock

import stomp

# Extend a basic listener class to accomodate custom message format

class CSVDataListener(stomp.ConnectionListener):
    # message_list = []

    def __init__(self):
        self.message_list = []
        self.error_list = []

    def on_error(self, headers, message):
        error_dict = {}
        error_dict['headers'] = headers
        error_dict['message'] = message

        current_datetime = datetime.now()
        current_timestamp = datetime.timestamp(current_datetime)

        error_dict['timestamp'] = current_timestamp
        self.error_list.append(error_dict)

        # Uncomment this line to print out a comment after every received error
        
        # print('Error received: "{}"'.format(message))
        
    def on_message(self, headers, message):
        # Parse received STOMP messages, serializing them into dictionaries

        raw = str(message)
        title, content = raw[:raw.find('{')], raw[raw.find('{')+1:raw.find('}')]
        fields = content.split(', ')
        result_dict = {}
        message_dict = {}

        for field in fields:
            key, value = field.split('=')
            key, value = key.strip(), value.strip()
            message_dict[key] = value
        
        result_dict['headers'] = headers
        result_dict['title'] = title
        result_dict['message'] = message_dict

        current_datetime = datetime.now()
        current_timestamp = datetime.timestamp(current_datetime)

        result_dict['timestamp'] = current_timestamp
        self.message_list.append(result_dict)
        # Uncomment this line to print out a comment after every received message
        
        # print('Message received: {}'.format(str(result_dict)))

if __name__ == "__main__":
    # If this script is launched directly (that is, using '$ python stomp_receiver.py' command), a STOMP listener will be created and a test message will be sent to it
    # Test message can be customized using command line arguments. If none provided, a basic placeholder will be used instead.
    
    if sys.argv[1:]:
        test_msg = sys.argv[1:]
    else:
        test_msg = "BookmarkMessage1{status=status, userId=userId, element=element, rate=rate, vol=vol, num=num, page=page, comment='comment'}"

    host_and_ports = [('0.0.0.0', 61613)]
    conn = stomp.Connection(host_and_ports=host_and_ports)
    
    base_listener = CSVDataListener()
    conn.set_listener('', base_listener)
    conn.start()
    conn.connect('admin', 'password', wait=True)

    conn.subscribe(destination='/queue/messages', id=1, ack='auto')
    conn.send(body=test_msg, destination='/queue/messages')
    
    time.sleep(1)
    messages = base_listener.message_list
    print(messages)
    conn.disconnect()
