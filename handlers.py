def handle_message(message, message_type):
    if message_type == 'element':
        return {'success': True, 'content': {'element_id': message.get('element', {}).get('id', None), 'user_id': message.get('userId', None), 'rating': message.get('weight', None), 'status': message.get('status', None)}}
    elif message_type == 'activity':
        return {'success': True, 'content': {'element_id': message.get('element', {}).get('id', None), 'title': message.get('name', None), 'tags': message.get('tagsNamesString', None)}}
    else:
        return {'success': False}