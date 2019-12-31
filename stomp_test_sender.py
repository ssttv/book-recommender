import stomp
import time

host_and_ports = [('0.0.0.0', 61613)]
conn = stomp.Connection(host_and_ports=host_and_ports)

conn.start()
conn.connect('admin', 'password', wait=True)
counter = 0

while counter <= 5:
    test_msg = "BookmarkMessage" + str(counter) + "{status=status, userId=userId, element=element, rate=rate, vol=vol, num=num, page=page, comment='comment'}"
    conn.send(body=test_msg, destination='/queue/messages')
    conn.send(body=test_msg, destination='/queue/messages')
    counter += 1
    time.sleep(0.1)

time.sleep(1)
# messages = base_listener.message_list
# print(messages)
conn.disconnect()
