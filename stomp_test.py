import time
import sys

import stomp

class MyListener(stomp.ConnectionListener):
    def on_error(self, headers, message):
        print('received an error "%s"' % message)
    def on_message(self, headers, message):
        print('received a message "%s"' % message)

class CSVDataListener(stomp.ConnectionListener):
    def on_error(self, headers, message):
        print('received an error "{}"'.format(message))
    def on_message(self, headers, message):
        raw = str(message)
        print(raw)
        title, content = raw[:raw.find('{')], raw[raw.find('{')+1:raw.find('}')]
        fields = content.split(',')
        out_dict = {}
        for field in fields:
            key, value = field.split('=')
            key, value = key.strip(), value.strip()
            out_dict[key] = value
        print('receved a {}: {}'.format(title, str(out_dict)))

if __name__ == "__main__":
    if sys.argv[1:]:
        test_msg = sys.argv[1:]
    else:
        test_msg = "BookmarkMessage1{status=status, userId=userId, element=element, rate=rate, vol=vol, num=num, page=page, comment='comment'}"

    hosts = [('0.0.0.0', 61613)]
    conn = stomp.Connection()
    conn.set_listener('', CSVDataListener())
    conn.start()
    conn.connect('admin', 'password', wait=True)

    conn.subscribe(destination='/queue/test', id=1, ack='auto')

    # conn.send(body=str(sys.argv[1:]), destination='/queue/test')
    conn.send(body=test_msg, destination='/queue/test')

    time.sleep(1)
    conn.disconnect()
